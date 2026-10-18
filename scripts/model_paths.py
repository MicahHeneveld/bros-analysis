#!/usr/bin/env python3
"""Locate a model checkpoint.

Weights are not in git. They reach a contributor through a `models-<run>` release
asset, downloaded by `bounty.py fetch --models` into `models/`. The maintainer's
machine also still has `.staging/runs/...`, and Colab has `<root>/runs/...`, so
lookup has to cover all three without any of them being mandatory.

Order matters: an explicit `--weights` always wins, then the maintainer/Colab
layout (where the newest local training run is the interesting one), then the
published model. That way publishing a checkpoint never silently overrides a run
someone is actively iterating on.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


class WeightsNotFound(FileNotFoundError):
    """Raised with a message aimed at a contributor, not a maintainer."""


def is_base_model(stem: str) -> bool:
    """True for an Ultralytics base checkpoint (`yolo11n`, `yolov8s`, …).

    Those land in the working directory just by running `YOLO("yolo11n.pt")`,
    and one sitting in models/ must never be mistaken for the player detector —
    it is an untrained general model and its output would look plausible while
    being completely wrong.
    """
    return stem.lower().startswith("yolo")


def _usable(paths: list[Path]) -> list[Path]:
    return [p for p in paths if not is_base_model(p.stem)]


def _newest(paths: list[Path]) -> Path | None:
    try:
        return max(paths, key=lambda p: p.stat().st_mtime)
    except (OSError, ValueError):
        return None


def candidate_paths(
    run_name: str | None = None,
    root: Path | None = None,
) -> list[Path]:
    """Every plausible checkpoint path, best maintained first."""
    out: list[Path] = []
    if run_name:
        if root is not None:
            out.append(root / "runs" / run_name / "weights" / "best.pt")
        out.append(REPO_ROOT / ".staging" / "runs" / run_name / "weights" / "best.pt")
        out.append(MODELS_DIR / f"{run_name}.pt")
    if root is not None:
        newest = _newest(list((root / "runs").glob("*/weights/best.pt")))
        if newest:
            out.append(newest)
    newest_staging = _newest(list((REPO_ROOT / ".staging" / "runs").glob("*/weights/best.pt")))
    if newest_staging:
        out.append(newest_staging)
    out.append(MODELS_DIR / "best.pt")
    newest_model = _newest(_usable(list(MODELS_DIR.glob("*.pt"))))
    if newest_model:
        out.append(newest_model)
    return out


def resolve_weights(
    explicit: Path | None = None,
    run_name: str | None = None,
    root: Path | None = None,
) -> Path:
    """Return the first existing checkpoint, or raise a helpful error."""
    if explicit is not None:
        explicit = explicit.expanduser()
        if explicit.exists():
            return explicit
        raise WeightsNotFound(
            f"--weights points at {explicit}, which does not exist."
        )

    seen: set[Path] = set()
    for path in candidate_paths(run_name, root):
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path

    tried = "\n".join(f"  {p}" for p in cardinal_lines(run_name, root))
    raise WeightsNotFound(
        "No model checkpoint found. Tried:\n"
        f"{tried}\n\n"
        "Get the published model with:\n"
        "  python scripts/bounty.py fetch --models\n"
        "or point at a checkpoint directly with --weights path/to/best.pt."
    )


def cardinal_lines(run_name: str | None, root: Path | None) -> list[Path]:
    """The two or three paths worth showing in an error message."""
    lines: list[Path] = []
    if run_name:
        lines.append(MODELS_DIR / f"{run_name}.pt")
    lines.append(MODELS_DIR / "best.pt")
    if root is not None:
        lines.append(root / "runs" / (run_name or "<run>") / "weights" / "best.pt")
    lines.append(REPO_ROOT / ".staging" / "runs" / (run_name or "<run>") / "weights" / "best.pt")
    return lines
