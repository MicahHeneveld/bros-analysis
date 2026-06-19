from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare YOLO prediction labels for CVAT import."
    )
    parser.add_argument(
        "src_labels",
        nargs="?",
        default=(
            "runs/detect/runs/ssbu_predictions/remaining_frames/labels"
        ),
        help="Folder containing YOLO .txt label files.",
    )
    parser.add_argument(
        "--out-dir",
        default="cvat_import_predictions",
        help="Output folder used to assemble the CVAT import package.",
    )
    parser.add_argument(
        "--zip-path",
        default="cvat_import_predictions.zip",
        help="Path for the final zip archive.",
    )
    parser.add_argument(
        "--class-name",
        default="player",
        help="Class name to write into data.yaml.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Class id to write into data.yaml.",
    )
    parser.add_argument(
        "--format",
        choices=("yolo11", "ultralytics"),
        default="yolo11",
        help="CVAT import package format to generate.",
    )
    return parser.parse_args()


def clean_label_lines(src_file: Path) -> list[str]:
    cleaned_lines: list[str] = []
    for raw_line in src_file.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        cleaned_lines.append(" ".join(parts[:5]))
    return cleaned_lines


def write_data_yaml(out_dir: Path, class_id: int, class_name: str) -> None:
    yaml_text = (
        "path: .\n"
        "train: images/default\n"
        "val: images/default\n\n"
        "nc: 1\n"
        "names:\n"
        f"  {class_id}: {class_name}\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def write_obj_names(out_dir: Path, class_name: str) -> None:
    (out_dir / "obj.names").write_text(f"{class_name}\n", encoding="utf-8")


def write_train_txt(out_dir: Path, label_names: list[str]) -> None:
    lines = [f"obj_train_data/{name.replace('.txt', '.jpg')}" for name in label_names]
    body = "\n".join(lines)
    if lines:
        body += "\n"
    (out_dir / "train.txt").write_text(body, encoding="utf-8")


def write_obj_data(out_dir: Path) -> None:
    body = (
        "classes = 1\n"
        "names = obj.names\n"
        "train = train.txt\n"
        "backup = backup/\n"
    )
    (out_dir / "obj.data").write_text(body, encoding="utf-8")


def zip_directory(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(source_dir))


def main() -> None:
    args = parse_args()

    src_labels = Path(args.src_labels)
    out_dir = Path(args.out_dir)
    out_labels = out_dir / "labels" / "default"
    out_yolo11_labels = out_dir / "obj_train_data"
    zip_path = Path(args.zip_path)

    if not src_labels.exists():
        raise FileNotFoundError(f"Source labels folder not found: {src_labels}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    if args.format == "ultralytics":
        out_labels.mkdir(parents=True, exist_ok=True)
    else:
        out_yolo11_labels.mkdir(parents=True, exist_ok=True)

    label_count = 0
    line_count = 0
    label_names: list[str] = []

    for src_file in sorted(src_labels.glob("*.txt")):
        cleaned_lines = clean_label_lines(src_file)
        if args.format == "ultralytics":
            dst_file = out_labels / src_file.name
        else:
            dst_file = out_yolo11_labels / src_file.name
        body = "\n".join(cleaned_lines)
        if cleaned_lines:
            body += "\n"
        dst_file.write_text(body, encoding="utf-8")
        label_count += 1
        line_count += len(cleaned_lines)
        label_names.append(src_file.name)

    if args.format == "ultralytics":
        write_data_yaml(out_dir, args.class_id, args.class_name)
    else:
        write_obj_names(out_dir, args.class_name)
        write_train_txt(out_dir, label_names)
        write_obj_data(out_dir)
    zip_directory(out_dir, zip_path)

    print(f"Source labels: {src_labels}")
    print(f"Cleaned label files: {label_count}")
    print(f"Total predicted boxes: {line_count}")
    print(f"Output folder: {out_dir.resolve()}")
    print(f"Wrote zip: {zip_path.resolve()}")


if __name__ == "__main__":
    main()
