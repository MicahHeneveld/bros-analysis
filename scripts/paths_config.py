"""Load Drive-canonical paths from configs/paths.yaml (or paths.example.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required. Install with: python -m pip install pyyaml"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "paths.yaml"
EXAMPLE_CONFIG_PATH = REPO_ROOT / "configs" / "paths.example.yaml"

DRIVE_FOLDER_ID = "12iT_dLsUOeFQ_UQBQsIWsnmD6ywvdPtl"
DRIVE_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"


def load_paths_config() -> dict[str, Any]:
    path = CONFIG_PATH if CONFIG_PATH.exists() else EXAMPLE_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Missing path config. Copy {EXAMPLE_CONFIG_PATH.name} to {CONFIG_PATH.name}."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("drive_folder_id", DRIVE_FOLDER_ID)
    data.setdefault("drive_url", DRIVE_URL)
    data.setdefault("drive_folder_name", "bros-analysis")
    data.setdefault("colab_root", "/content/drive/MyDrive/bros-analysis")
    data.setdefault("dataset_dirname", "ssbu_players_colab")
    data.setdefault("runs_dirname", "runs")
    data.setdefault("frames_dirname", "frames")
    data.setdefault("exports_dirname", "exports/cvat")
    return data


def local_root(cfg: dict[str, Any] | None = None) -> Path | None:
    cfg = cfg or load_paths_config()
    raw = (cfg.get("local_root") or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def require_local_root(cfg: dict[str, Any] | None = None) -> Path:
    root = local_root(cfg)
    if root is None:
        raise SystemExit(
            "configs/paths.yaml has an empty local_root.\n"
            "1) Open the shared Drive folder and use Organize → Add shortcut to My Drive.\n"
            f"   {DRIVE_URL}\n"
            "2) Install Google Drive for desktop and wait for sync.\n"
            "3) Set local_root in configs/paths.yaml to the synced bros-analysis path.\n"
            "   See configs/paths.example.yaml."
        )
    return root


def dataset_dir(root: Path, cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_paths_config()
    return root / cfg["dataset_dirname"]


def runs_dir(root: Path, cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_paths_config()
    return root / cfg["runs_dirname"]


def frames_dir(root: Path, cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_paths_config()
    return root / cfg["frames_dirname"]


def exports_dir(root: Path, cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_paths_config()
    return root / cfg["exports_dirname"]
