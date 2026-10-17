# Distance between fighters

Measuring how far apart two characters are sounds trivial and is not. SSBU's
camera frames both fighters, so it zooms out as they separate. A measurement in
raw pixels is therefore not monotonic with true separation: a tight close-up and
a wide stage can produce the same pixel gap.

This document records what works, what does not, and the evidence.

## The zoom problem, measured

On this project's data (2,282 two-fighter frames):

| quantity | value |
|---|---|
| Apparent fighter height | 6.5% – 73% of frame height |
| Implied zoom range | **11.3x** |

Any pixel-space measurement is wrong by up to an order of magnitude.

## What works: body-lengths

Because apparent height is proportional to \(1/\text{depth}\), dividing the
separation by it cancels the zoom:

\[
d = \frac{\lVert c_1 - c_2 \rVert}{\tfrac{1}{2}(h_1 + h_2)}
\]

"No calibration, no stage known, works on any footage." Shot-to-shot zoom is
divided out entirely.

**Validated.** Constructing a synthetic perspective camera with a known
ground plane, horizon and camera height, `body_lengths` recovers the true
separation exactly when both fighters are at the same depth (0.0% error), and
degrades predictably as depth separation grows.

Measured on the labelled val split (427 frames with exactly two fighters):

| metric | mean | median | p05 | p95 | max |
|---|---|---|---|---|---|
| body_lengths | 2.27 | 1.96 | 0.72 | 4.84 | 8.25 |
| body_lengths, smoothed (window 5) | 2.10 | 1.98 | 1.08 | 3.63 | 4.86 |

~2 body-lengths apart is a plausible typical neutral spacing, which is a
useful sanity check.

**Where it fails.** Body-lengths is blind to *depth* separation. Two fighters
one directly in front of the other read as almost touching. On the synthetic
camera, pure depth separation (same X, Z 10 → 15) gives 0.80 body-lengths
against a true 5.0 — an 84% error. Across mixed-depth pairs the median error is
33%.

Smoothing helps because bbox height wobbles with pose, crouching, hitstun and
character size; a rolling median over neighbouring frames in the same shot
removes much of that. Note the p95 drop (4.84 → 3.63), which is the smoothing
suppressing spurious spikes rather than a real change in the distribution.

## What does not work: ground-plane inversion

The obvious next step is to recover depth properly. For a camera at height \(H\)
with focal length \(f\), and a fighter of world height \(S\) on a common floor:

\[
\frac{h_{\text{img}}}{H_{\text{frame}}} = \frac{fS}{Z}, \qquad
\frac{y_{\text{feet}}}{H_{\text{frame}}} - y_{\text{horizon}} = \frac{fH}{Z}
\]

Dividing, the unknown depth cancels, leaving a straight line:

\[
y_{\text{feet}} - y_{\text{horizon}} = \frac{H}{S}\,h_{\text{img}}
\]

So the horizon and camera height are the **intercept and slope fitting feet-Y
against apparent height** — a self-calibration from the detections alone, no
stage geometry or manual annotation. `app/distance.py` implements this and
recovers a synthetic camera's horizon and camera height to **0.0000 error**,
then reproduces distances to **0.00%**.

**It does not work on real SSBU footage.** The fitted line is flat:

| test | result |
|---|---|
| Whole val split (n=612 points) | R² = **0.01** |
| Single match, fixed matchup (n=292) | R² = **0.00** |
| Best single 30-frame window | R² = **0.02** |
| Correlation of Δheight with Δfeet-Y | **+0.06** |

The model predicts a strong positive correlation. Real data shows none, at any
time scale — so this is not the camera moving between shots. Testing short
windows separately isolates that: the fit is equally bad from 30 frames to the
whole match, which points at the height signal itself, not camera drift.

### Why it fails

1. **Character identity dominates apparent height.** Bowser is roughly twice
   Pikachu's height at identical depth. The model assumes one constant \(S\),
   and this footage has varied matchups.
2. **Pose dominates too.** Crouching, attacking, being launched and ledge-hanging
   all change bbox height by large factors independent of depth.
3. **SSBU stages are shallow.** Fighters occupy a narrow vertical band
   (feet_y p05 = 0.29, p95 = 0.88 of frame height, but clustered near the
   platform), so genuine depth separation is small and the depth term is mostly
   noise.

The practical consequence is reassuring: **because depth separation is small,
body-lengths is close to correct almost all the time**, and the correction it
cannot provide is correcting for something that barely varies.

## Recommended usage

```bash
python scripts/measure_distance.py .staging/ingested/<video>
```

Writes `distance.csv` (per frame) and `distance.summary.json`.

Use `body_smoothed` for anything downstream — rates of change, average spacing,
approach/retreat. Use raw `body_lengths` only when you need single-frame values
and understand the pose noise.

The ground-plane metric is available behind `--ground` for research. It is
**off by default deliberately**: it produces confident-looking numbers from a
fit with R² ≈ 0, which is worse than not offering it.

## What would actually improve this

In rough order of value:

1. **Temporal tracking.** Gives each fighter an identity across frames, which
   unlocks everything below. Ultralytics ships ByteTrack/BoT-SORT.
2. **Per-character scale calibration.** With identity, take each character's
   95th-percentile observed height across a match as their canonical size at
   closest approach. Normalising by that removes the Bowser/Pikachu confound
   that defeats the ground-plane model — this is the single highest-value fix.
3. **Pose-aware height.** Use the bbox of the *torso* or the head-to-foot span
   rather than the whole box, so crouching and attacks stop corrupting depth.
4. **Stage-aware homography.** Detect the main platform, calibrate against its
   known dimensions. Accurate but stage-specific, and a lot of work.
5. **Sanity check against percent HUD.** Not a distance source, but the damage
   readout can corroborate a hit event, which validates a spacing measurement
   independently.

Until (1) and (2) exist, body-lengths is the honest answer.
