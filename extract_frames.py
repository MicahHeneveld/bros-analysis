import argparse
from pathlib import Path
import sys

try:
    import cv2
except Exception as e:
    print("ERROR: OpenCV (cv2) is not installed or failed to import:", e)
    print("Install with: python -m pip install opencv-python")
    sys.exit(1)


def extract_frames(video_path: Path, output_dir: Path, every_n_frames: int = 15):
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


def main():
    p = argparse.ArgumentParser(description="Extract frames from a video file.")
    p.add_argument("video", nargs="?", default="data/raw_videos/YTMP3GG_YouTube_UAS-Finale-GRAND-FINALS-MkLeo-Pyra-Mythr_Media_hI9gPuGN07c_001_1080p.mp4",
                   help="Path to input video file")
    p.add_argument("-o", "--outdir", help="Output directory (defaults to data/frames/<video_stem>)")
    p.add_argument("-n", "--every", type=int, default=15, help="Save every Nth frame (default: 15)")

    args = p.parse_args()
    video_path = Path(args.video)

    if not video_path.exists():
        print(f"ERROR: video file not found: {video_path}")
        sys.exit(2)

    outdir = Path(args.outdir) if args.outdir else Path("data") / "frames" / video_path.stem

    try:
        saved = extract_frames(video_path, outdir, every_n_frames=args.every)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(3)

    print(f"Saved {saved} frames to {outdir}")


if __name__ == "__main__":
    main()