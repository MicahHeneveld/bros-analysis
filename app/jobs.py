"""Background jobs for "Add dataset".

Adding a dataset means running ``scripts/ingest_video.py`` over an uploaded MP4,
which takes minutes on a real match. Doing that inside an HTTP request would
block the landing screen and time out in the browser, so it runs in a worker
thread and the client polls ``/api/jobs/<id>``.

Only one ingest runs at a time: they are CPU/GPU bound and two would compete for
the same device while both reporting half-speed progress.

The subprocess is chosen deliberately. The review server itself runs on the
stdlib-only venv, while ultralytics lives in a training venv, so the job cannot
assume ``sys.executable`` can import it. Resolution order is an explicit env
override, then ``.venv-train``, then the server's own interpreter.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

# `  120 sampled -> 48 kept` from ingest_video.py's periodic progress line.
PROGRESS_RE = re.compile(r"(\d+)\s+sampled\s+->\s+(\d+)\s+kept")

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"


def ingest_python(repo_root: Path) -> Path:
    """Interpreter that can import ultralytics."""
    override = os.environ.get("BROS_INGEST_PYTHON", "").strip()
    if override:
        return Path(override).expanduser()
    for candidate in (
        repo_root / ".venv-train" / "bin" / "python",
        repo_root / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def probe_video(python: Path, video: Path, timeout: float = 30.0) -> dict:
    """Duration and size, read with the ingest interpreter's OpenCV.

    Used only to turn "N frames sampled" into a percentage. A probe failure is
    not fatal: progress falls back to an indeterminate bar rather than claiming
    a percentage it cannot compute.
    """
    script = (
        "import json,sys\n"
        "import cv2\n"
        "c=cv2.VideoCapture(sys.argv[1])\n"
        "print(json.dumps({'fps': c.get(cv2.CAP_PROP_FPS) or 0,\n"
        " 'frames': c.get(cv2.CAP_PROP_FRAME_COUNT) or 0,\n"
        " 'w': c.get(cv2.CAP_PROP_FRAME_WIDTH) or 0,\n"
        " 'h': c.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0}))\n"
        "c.release()\n"
    )
    try:
        res = subprocess.run(
            [str(python), "-B", "-c", script, str(video)],
            capture_output=True, text=True, timeout=timeout,
        )
        if res.returncode == 0:
            data = json.loads(res.stdout.strip().splitlines()[-1])
            fps = float(data.get("fps") or 0)
            frames = float(data.get("frames") or 0)
            return {
                "fps": fps,
                "frames": int(frames),
                "duration": round(frames / fps, 2) if fps else 0.0,
                "width": int(data.get("w") or 0),
                "height": int(data.get("h") or 0),
            }
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        pass
    return {}


class Job:
    def __init__(self, kind: str, label: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.label = label
        self.status = STATUS_QUEUED
        self.percent = 0.0
        self.message = "queued"
        self.lines: list[str] = []
        self.error = ""
        self.warnings: list[str] = []
        self.created = time.time()
        self.finished = 0.0
        self.result: dict = {}
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def log(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        with self._lock:
            self.lines.append(line)
            # Keep a bounded tail: a long ingest can print thousands of lines and
            # the client only ever shows the last few.
            if len(self.lines) > 400:
                del self.lines[:200]

    def set(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def as_dict(self, tail: int = 12) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "label": self.label,
                "status": self.status,
                "percent": round(self.percent, 1),
                "message": self.message,
                "error": self.error,
                "warnings": list(self.warnings),
                "result": dict(self.result),
                "created": self.created,
                "finished": self.finished,
                "log": self.lines[-tail:],
            }


class JobRunner:
    def __init__(self, repo_root: Path, board, on_dataset_ready=None):
        self.repo_root = Path(repo_root).resolve()
        self.board = board
        self.on_dataset_ready = on_dataset_ready
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._active: str = ""

    # ----- queries -------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def active(self) -> dict | None:
        """The running job, if any.

        A finished job must not look active: the landing screen polls this on
        load, and a stale entry would replay an old ingest (and its dataset
        name) over whatever the user just opened.
        """
        with self._lock:
            if not self._active:
                return None
            job = self._jobs.get(self._active)
            if job is None or job.status not in (STATUS_QUEUED, STATUS_RUNNING):
                return None
            return job.as_dict()

    def _clear_active(self, job_id: str) -> None:
        with self._lock:
            if self._active == job_id:
                self._active = ""

    def recent(self, limit: int = 5) -> list[dict]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)
        return [j.as_dict() for j in jobs[:limit]]

    # ----- ingest --------------------------------------------------------
    def start_ingest(
        self,
        video: Path,
        name: str,
        *,
        fps: float = 5.0,
        publish: bool = False,
        target: int = 250,
        cap: int = 6,
    ) -> Job:
        with self._lock:
            if self._active:
                current = self._jobs.get(self._active)
                if current and current.status in (STATUS_QUEUED, STATUS_RUNNING):
                    raise RuntimeError("an ingest is already running")
            job = Job("ingest", name)
            self._jobs[job.id] = job
            self._active = job.id

        thread = threading.Thread(
            target=self._run_ingest,
            args=(job, Path(video), name, fps, publish, target, cap),
            daemon=True,
        )
        thread.start()
        return job

    def _run_ingest(
        self,
        job: Job,
        video: Path,
        name: str,
        fps: float,
        publish: bool,
        target: int,
        cap: int,
    ) -> None:
        python = ingest_python(self.repo_root)
        dataset = self.repo_root / "data" / "datasets" / name
        try:
            info = probe_video(python, video)
            expected = (info.get("duration") or 0) * fps if info else 0

            cmd = [
                str(python), "-B", "scripts/ingest_video.py", str(video),
                "-o", str(dataset), "--split", name, "--fps", str(fps),
            ]
            job.set(status=STATUS_RUNNING, message="starting detector…")
            job.log("$ " + " ".join(cmd))

            proc = subprocess.Popen(
                cmd,
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            job.proc = proc

            for line in proc.stdout or []:
                job.log(line)
                match = PROGRESS_RE.search(line)
                if match:
                    sampled = int(match.group(1))
                    kept = int(match.group(2))
                    percent = min(99.0, 100.0 * sampled / expected) if expected else 0.0
                    job.set(
                        percent=percent,
                        message=f"{sampled} frames sampled · {kept} kept",
                    )
                elif line.startswith(("Video", "Sampling", "Device", "Gating")):
                    job.set(message=line.strip())

            code = proc.wait()
            if job.status == STATUS_CANCELLED:
                return
            if code != 0:
                job.set(status=STATUS_ERROR, error=f"ingest exited with code {code}",
                        message="ingest failed", finished=time.time())
                return

            job.set(percent=99.0, message="indexing labels…")
            images = sum(
                1 for split in (dataset / "images").glob("*") if split.is_dir()
                for p in split.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
            )
            if images == 0:
                job.set(
                    status=STATUS_ERROR,
                    error="no frames were kept — the clip may not be gameplay",
                    message="no frames kept",
                    finished=time.time(),
                )
                return

            job.result["dataset"] = name
            job.result["images"] = images

            job.set(message="creating bounties…")
            created = self.board.seed(name, target=target, cap=cap)
            job.result["bounties"] = len(created)

            if publish:
                self._publish(job, name, dataset, created)

            job.set(status=STATUS_DONE, percent=100.0, message="ready to review",
                    finished=time.time())
            if self.on_dataset_ready:
                try:
                    self.on_dataset_ready(name)
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001
            if job.status != STATUS_CANCELLED:
                job.set(status=STATUS_ERROR, error=str(exc).splitlines()[0],
                        message="failed", finished=time.time())
        finally:
            self._clear_active(job.id)

    def _publish(self, job: Job, name: str, dataset: Path, created: list[dict]) -> None:
        """Best-effort GitHub steps, reported per step.

        Neither step aborts the job: the dataset and its local bounties are
        already useful, and losing them because GitHub was unreachable would be
        a worse trade than leaving publishing to a later CLI run.
        """
        python = self.repo_root / ".venv" / "bin" / "python"
        python = python if python.exists() else Path(sys.executable)

        job.set(message="opening GitHub issues…")
        for bounty in created:
            cmd = [
                str(python), "-B", "scripts/bounty.py", "create",
                "--dataset", str(dataset),
                "--target", str(bounty.get("target") or 250),
            ]
            try:
                res = subprocess.run(
                    cmd, cwd=self.repo_root, capture_output=True, text=True, timeout=180,
                )
                out = (res.stdout or "").strip()
                if res.returncode != 0:
                    job.warnings.append(
                        f"issue for {bounty['id']}: {(res.stderr or out or 'failed').strip().splitlines()[0]}"
                    )
                    continue
                url = next(
                    (line.split("Opened bounty:", 1)[1].strip()
                     for line in out.splitlines() if "Opened bounty:" in line),
                    "",
                )
                if url:
                    number = int(url.rstrip("/").rsplit("/", 1)[-1])
                    self.board.link_github(bounty["id"], number, url)
                else:
                    job.warnings.append(f"issue for {bounty['id']}: no URL returned")
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                job.warnings.append(f"issue for {bounty['id']}: {exc}")
                break

        # Image bundles are what let other people review the dataset; without
        # one, only this machine can open it.
        job.set(message="publishing image bundle…")
        bundle = self.repo_root / ".staging" / "bundles" / f"{name}.zip"
        try:
            bundle.parent.mkdir(parents=True, exist_ok=True)
            shutil.make_archive(str(bundle.with_suffix("")), "zip", root_dir=dataset, base_dir="images")
            res = subprocess.run(
                ["gh", "release", "create", f"images-{name}", str(bundle),
                 "--title", f"{name} images",
                 "--notes", f"Frame bundle for `{name}`, produced by the review app's Add dataset flow."],
                cwd=self.repo_root, capture_output=True, text=True, timeout=1800,
            )
            if res.returncode != 0:
                job.warnings.append(
                    "image release: " + (res.stderr or res.stdout or "failed").strip().splitlines()[0]
                )
            else:
                job.result["release"] = res.stdout.strip()
        except (subprocess.SubprocessError, OSError) as exc:
            job.warnings.append(f"image release: {exc}")

    # ----- control -------------------------------------------------------
    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.status not in (STATUS_QUEUED, STATUS_RUNNING):
            return False
        job.set(status=STATUS_CANCELLED, message="cancelled", finished=time.time())
        proc = job.proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
        return True


def slugify(text: str) -> str:
    """Dataset name from a filename: same rules ingest_video.py uses."""
    out = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return out[:64] or "dataset"


def unique_name(repo_root: Path, base: str) -> str:
    """A dataset name that does not collide with an existing directory."""
    candidate = base
    n = 2
    while (repo_root / "data" / "datasets" / candidate).exists() or (
        repo_root / ".staging" / "ingested" / candidate
    ).exists():
        candidate = f"{base}_{n}"
        n += 1
    return candidate
