"""Distance between two fighters, corrected for SSBU's dynamic camera.

The problem
-----------
SSBU's camera frames both fighters, so it zooms out as they separate. Measured
in raw pixels the distance is therefore not monotonic with true separation:
a wide stage and a tight close-up can produce the same pixel gap. On this
project's data the apparent fighter height spans 6.5%..73% of frame height,
an ~11x zoom range, so pixel distances are wrong by up to an order of magnitude.

Because fighter height is proportional to 1/depth, dividing by it cancels the
zoom. That is the ``body`` metric below.

Two metrics, two levels of cost
-------------------------------
``body``
    Separation divided by mean apparent fighter height: "how many body-lengths
    apart". No calibration, no stage known, works on any footage. Exact when
    both fighters are at the same depth, which is the common case; degrades as
    depth separation grows, because the two fighters then have different scales.

``ground``
    Assumes both fighters stand on a common ground plane and inverts the
    perspective projection. This recovers the *depth* separation that ``body``
    cannot see, at the cost of a noise-sensitive term.

The ground-plane model
----------------------
With a camera of vertical field of view giving ``g = f / frame_height``
(``g = 1 / (2 tan(vfov/2))``) at height ``H`` above the floor, and a fighter of
world height ``S`` at world position ``(X, Z)``:

    apparent_height / frame_height  =  g * S / Z
    feet_y / frame_height          - horizon =  g * H / Z

Dividing, the unknown depth cancels:

    feet_y_norm - horizon_norm  =  (H / S) * height_norm

So the horizon and the camera height (in fighter-heights) are the *intercept and
slope of a straight line* fitting feet-Y against apparent height. That is a
self-calibration from the detections alone -- no stage geometry, no manual
annotation -- as long as fighters are on the floor when they are sampled.

Inverting the projection then gives positions in units of a fighter's height:

    X / S = x_norm / height_norm
    Z / S = g / height_norm

and the distance follows as the norm of the difference of those two vectors.
Noise warning: the Z term is a difference of reciprocals of a noisy measurement
(bbox height varies with pose, crouching, hitstun and character size), so it
amplifies error far more than the X term. Callers should prefer ``body`` unless
depth separation matters and should watch ``depth_ratio`` to know when the two
metrics disagree meaningfully.

Boxes are YOLO normalized ``cxcywh`` dicts, matching ``predictions.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default for g = 1 / (2 tan(vfov/2)). SSBU at 16:9 is roughly a 45-degree
# vertical field of view, giving ~1.2. This is the least certain constant in the
# model: it only scales the depth term, so an error here mis-weights depth
# separation against lateral separation, it does not corrupt either term alone.
DEFAULT_G = 1.2

# Below this many calibrated samples a fitted horizon is not worth trusting.
MIN_CALIBRATION_SAMPLES = 40


@dataclass
class Calibration:
    """Horizon and camera height recovered from feet-Y vs apparent height."""

    horizon_norm: float
    """Horizon line as a fraction of frame height, measured from the top."""

    camera_height: float
    """Camera height above the floor, in fighter-heights."""

    r2: float
    samples: int
    reliable: bool
    note: str = ""


@dataclass
class Measurement:
    """Distance between two fighters in one frame."""

    body_lengths: float
    """Separation divided by mean apparent height. Always available."""

    depth_ratio: float
    """Taller/shorter apparent height. 1.0 means same depth and character size."""

    ground_lengths: float | None = None
    """Ground-plane distance in fighter-heights. None without a calibration."""

    lateral: float | None = None
    depth: float | None = None
    """The two components of ``ground_lengths``, so a caller can see which
    direction dominates. Large ``depth`` with small ``lateral`` is exactly the
    case ``body_lengths`` misreports."""

    def as_dict(self) -> dict:
        out = {
            "body_lengths": round(self.body_lengths, 4),
            "depth_ratio": round(self.depth_ratio, 4),
        }
        if self.ground_lengths is not None:
            out["ground_lengths"] = round(self.ground_lengths, 4)
            out["lateral"] = round(self.lateral or 0.0, 4)
            out["depth"] = round(self.depth or 0.0, 4)
        return out


def feet_and_size(box: dict, width: float, height: float) -> tuple[float, float, float, float]:
    """``(x_feet_px, y_feet_px, box_h_px, box_w_px)`` for one detection.

    The feet -- bottom-centre -- are the ground-contact point, which is the
    geometrically meaningful one for a ground-plane model. Using the box centre
    would shift with the character's pose.
    """
    bw = float(box["w"]) * width
    bh = float(box["h"]) * height
    x = float(box["xc"]) * width
    y = (float(box["yc"]) + float(box["h"]) / 2.0) * height
    return x, y, bh, bw


def body_lengths(a: dict, b: dict, width: float, height: float) -> tuple[float, float]:
    """Scale-free separation and the ratio of the two apparent heights.

    Center-to-center, because the centre is more pose-stable than the feet and
    this metric makes no ground-plane assumption.

    Returns ``(body_lengths, depth_ratio)``.
    """
    ax = float(a["xc"]) * width
    ay = float(a["yc"]) * height
    bx = float(b["xc"]) * width
    by = float(b["yc"]) * height
    dx, dy = ax - bx, ay - by
    dist = (dx * dx + dy * dy) ** 0.5

    ah = float(a["h"]) * height
    bh = float(b["h"]) * height
    mean_h = (ah + bh) / 2.0
    if mean_h <= 0:
        return 0.0, 1.0
    ratio = (max(ah, bh) / min(ah, bh)) if min(ah, bh) > 0 else 1.0
    return dist / mean_h, ratio


def calibrate(
    pairs: list[tuple[dict, dict]],
    width: float,
    height: float,
    *,
    max_depth_ratio: float = 1.35,
) -> Calibration | None:
    """Fit the horizon from feet-Y against apparent height.

    Only fighters whose two heights are close are used: a large height ratio
    means different depth or different characters, and pooling those biases the
    fit. That is the same subset on which ``body`` is already exact, so the
    calibration is learned where it can be trusted.
    """
    xs: list[float] = []
    ys: list[float] = []
    for a, b in pairs:
        ah = float(a["h"])
        bh = float(b["h"])
        if min(ah, bh) <= 0:
            continue
        if max(ah, bh) / min(ah, bh) > max_depth_ratio:
            continue
        for box in (a, b):
            _, y, bh_px, _ = feet_and_size(box, width, height)
            xs.append(bh_px / height)
            ys.append(y / height)

    n = len(xs)
    if n < MIN_CALIBRATION_SAMPLES:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # A physically sensible fit needs a positive slope (higher on screen means
    # further away, so a shorter fighter) and a horizon inside, or just above,
    # the frame. Slope <= 0 means the assumption is simply not holding.
    reliable = slope > 0 and -0.2 <= intercept <= 1.2 and r2 >= 0.35
    note = ""
    if slope <= 0:
        note = "feet-Y does not fall with apparent height; ground plane assumption invalid"
    elif not (-0.2 <= intercept <= 1.2):
        note = f"horizon {intercept:.2f} outside a plausible range"
    elif r2 < 0.35:
        note = f"weak fit (R2={r2:.2f}); lots of airborne or off-plane fighters"
    return Calibration(
        horizon_norm=intercept,
        camera_height=slope,
        r2=r2,
        samples=n,
        reliable=reliable,
        note=note,
    )


def measure(
    a: dict,
    b: dict,
    width: float,
    height: float,
    calibration: Calibration | None = None,
    *,
    g: float = DEFAULT_G,
) -> Measurement:
    """Distance between two fighters, both ways.

    ``ground_lengths`` is only populated when a *reliable* calibration is
    supplied; a fit with a bad horizon would produce confidently wrong numbers,
    which is worse than returning only the scale-free metric.
    """
    body, ratio = body_lengths(a, b, width, height)
    out = Measurement(body_lengths=body, depth_ratio=ratio)

    if calibration is None or not calibration.reliable:
        return out

    xa, _, ha, _ = feet_and_size(a, width, height)
    xb, _, hb, _ = feet_and_size(b, width, height)
    if min(ha, hb) <= 0:
        return out

    # Normalize by frame height so x and h share units. x is taken relative to
    # the optical axis, which the model assumes is the frame centre.
    cx = width / 2.0
    xa_n = (xa - cx) / height
    xb_n = (xb - cx) / height
    ha_n = ha / height
    hb_n = hb / height

    lateral = xa_n / ha_n - xb_n / hb_n
    depth = g * (1.0 / ha_n - 1.0 / hb_n)

    out.lateral = abs(lateral)
    out.depth = abs(depth)
    out.ground_lengths = (lateral * lateral + depth * depth) ** 0.5
    return out


def summarize(values: list[float]) -> dict:
    """Mean/median/percentiles for a list of distances."""
    if not values:
        return {"n": 0}
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return s[idx]

    return {
        "n": n,
        "mean": round(sum(s) / n, 3),
        "median": round(pct(0.5), 3),
        "p05": round(pct(0.05), 3),
        "p95": round(pct(0.95), 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
    }
