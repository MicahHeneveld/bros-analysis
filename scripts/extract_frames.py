"""Extract frames from a video into the Drive-canonical frames/ tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import cv2
except Exception as e:
    print("ERROR: OpenCV (cv2) is not installed or failed to import:", e)
    print("Install with: python -m pip install opencv-python")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths_config import frames_dir, load_paths_config, local_root


def extract_frames(video_path: Path, output_dir: Path, every_n_frames: int = 15) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            out_path = output_dir / f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def default_outdir(video_path: Path) -> Path:
    cfg = load_paths_config()
    root = local_root(cfg)
    if root is not None:
        return frames_dir(root, cfg) / video_path.stem
    return Path("data") / "frames" / video_path.stem


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract frames from a video file into Drive frames/ (or data/frames/)."
    )
    p.add_argument(
        "video",
        nargs="?",
        default=(
            "data/raw_videos/YTMP3GG_YouTube_UAS-Finale-GRAND-FINALS-"
            "MkLeo-Pyra-Mythr_Media_hI9gPuGN07c_001_1080p.mp4"
        ),
        help="Path to input video file",
    )
    p.add_argument(
        "-o",
        "--outdir",
        help="Output directory (default: <local_root>/frames/<video_stem> or data/frames/<stem>)",
    )
    p.add_argument(
        "-n",
        "--every",
        type=int,
        default=15,
        help="Save every Nth frame (default: 15)",
    )

    args = p.parse_args()
    video_path = Path(args.video)

    if not video_path.exists():
        print(f"ERROR: video file not found: {video_path}")
        sys.exit(2)

    outdir = Path(args.outdir) if args.outdir else default_outdir(video_path)

    try:
        saved = extract_frames(video_path, outdir, every_n_frames=args.every)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(3)

    print(f"Saved {saved} frames to {outdir}")


if __name__ == "__main__":
    main()
