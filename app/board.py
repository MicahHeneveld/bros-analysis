"""Dataset catalogue and bounty board for the review app's landing screen.

Two responsibilities that belong together because the landing screen needs both
at once:

* **Datasets** - which datasets exist and how far along their labelling is. A
  dataset is either *published* (``data/datasets/<name>``, labels in git) or
  *scratch* (``.staging/ingested/<name>``, throwaway work from an ingest).

* **Bounties** - named slices of a dataset with a reward attached. The shared
  record of a bounty is a GitHub Issue (see ``scripts/bounty.py``), but the
  landing screen has to work without ``gh`` installed, offline, or before any
  issue exists. So this module keeps a local mirror in ``.review/board.json``
  and treats GitHub as a sync target, not a dependency.

Why slices are stored locally
-----------------------------
``scripts/bounty.py`` defines a bounty as a *filter plus a target count*, which
is right for GitHub: two people can claim overlapping bounties and the app's
"unreviewed first" ordering keeps them from colliding. That model cannot answer
"how much of *this* bounty is done", because there is no list of frames.

For the landing screen to draw a truthful progress ring, a bounty needs its own
frame list. Bounties created here therefore carry ``slice`` - an explicit,
non-overlapping partition of the unreviewed frames. Bounties imported from
GitHub without a slice fall back to counting reviewed frames inside their
filter, capped at the target, and the UI labels that as an estimate.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_TARGET = 250
MAX_BOUNTIES = 6

# Same contract as scripts/bounty.py: an HTML comment GitHub does not render.
META_RE = re.compile(r"<!--\s*bounty:\s*(\{.*?\})\s*-->", re.S)
CLAIM_RE = re.compile(r"^[ \t]*claim by @([A-Za-z0-9-]+)[ \t]*$", re.I | re.M)
# `scripts/bounty.py create --count N` stamps the issue title with `[n/N]`.
# Together with the dataset in the metadata it identifies which seeded bounty
# an issue is about, which is how a local bounty adopts its issue on sync.
TITLE_ORDINAL_RE = re.compile(r"\[(\d+)/(\d+)\]\s*$")

STATUS_OPEN = "open"
STATUS_CLAIMED = "claimed"
STATUS_DONE = "done"

BOUNTY_LABEL = "bounty"
LABEL_CLAIMED = "bounty:claimed"
LABEL_DONE = "bounty:done"

# Why a GitHub call failed, in terms the landing screen can act on. `gh` reports
# everything as one opaque line, so the interesting split has to be recovered
# from the message. `blocked` is deliberately distinct from `auth`: a 403 from
# an intermediary (proxy, sandbox, abuse filter) looks nothing like a bad token,
# and telling someone to re-authenticate when their token is fine wastes time.
GH_HINTS = {
    "blocked": (
        "GitHub refused the request (403). Your token may be fine and the "
        "network path blocked instead: retry, and if it keeps failing restart "
        "the server, then check `gh auth status`."
    ),
    "auth": "GitHub did not accept the credentials. Run `gh auth login`, then retry.",
    "network": "No route to GitHub. Check the connection, then retry.",
    "unknown": "Retry, or run `gh issue list --label bounty` to see the raw error.",
}


def gh_failure(exc: object) -> dict:
    """Classify a failed ``gh`` call into a kind plus a hint worth showing.

    The kinds are ordered so the first matching pattern wins; ``blocked`` is
    checked before ``auth`` because a 403 body often mentions both.
    """
    raw = str(exc or "").strip()
    message = raw.splitlines()[0] if raw else "gh failed"
    low = message.lower()
    if "403" in message or "forbidden" in low or "rate limit" in low:
        kind = "blocked"
    elif "401" in message or "bad credentials" in low or "auth" in low:
        kind = "auth"
    elif any(token in low for token in (
        "timeout", "timed out", "connection", "no such host",
        "could not resolve", "network", "dial tcp", "eof",
    )):
        kind = "network"
    else:
        kind = "unknown"
    return {"kind": kind, "error": message, "hint": GH_HINTS[kind]}


# ------------------------------------------------------------------ helpers
def _atomic_json(path: Path, payload: dict) -> None:
    """Write JSON so an interrupted write cannot truncate the board."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _issue_ordinal(title: object) -> int:
    """The `n` in a bounty title's `[n/N]` suffix, or 0 when there is none."""
    match = TITLE_ORDINAL_RE.search(str(title or ""))
    return int(match.group(1)) if match else 0


def dump_meta(meta: dict) -> str:
    """Same contract as ``scripts/bounty.py``: a comment GitHub does not render."""
    return "<!-- bounty: " + json.dumps(meta, sort_keys=True) + " -->"


def _has_boxes(label_path: Path) -> bool:
    """A label file with at least one non-blank line.

    An *empty* label file is a deliberate "reviewed, no players" marker, so it
    counts as labelled work but not as a boxed frame.
    """
    try:
        with open(label_path, "r", encoding="utf-8") as fh:
            return any(line.strip() for line in fh)
    except OSError:
        return False


def _scan_dataset(root: Path) -> dict:
    """Counts for one dataset, scanning images and labels on disk.

    Deliberately disk-truth rather than cache: the landing screen is the one
    place that has to be right about how much work is left, and a stale cache
    there would mislead every contributor at once. A few thousand stat calls is
    a few milliseconds.
    """
    root = Path(root)
    images_dir = root / "images"
    labels_dir = root / "labels"
    review_path = root / ".review" / "review_state.json"

    reviewed_ids: set[str] = set()
    try:
        data = json.loads(review_path.read_text(encoding="utf-8"))
        entries = data.get("images", data) if isinstance(data, dict) else {}
        if isinstance(entries, dict):
            reviewed_ids = {
                k for k, v in entries.items()
                if isinstance(v, dict) and v.get("status") not in (None, "unreviewed")
            }
    except (OSError, ValueError):
        reviewed_ids = set()

    splits: dict[str, dict] = {}
    total = 0
    labelled = 0
    boxed = 0
    seen: list[str] = []

    if images_dir.is_dir():
        for split_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
            split = split_dir.name
            counts = splits.setdefault(split, {"images": 0, "labelled": 0, "boxed": 0})
            for image in sorted(split_dir.iterdir()):
                if not _is_image(image):
                    continue
                image_id = f"{split}/{image.stem}"
                seen.append(image_id)
                total += 1
                counts["images"] += 1
                label = labels_dir / split / f"{image.stem}.txt"
                if not label.exists():
                    # Some hand-built datasets keep a flat labels/ dir.
                    flat = labels_dir / f"{image.stem}.txt"
                    if flat.exists():
                        label = flat
                    else:
                        continue
                labelled += 1
                counts["labelled"] += 1
                if _has_boxes(label):
                    boxed += 1
                    counts["boxed"] += 1

    reviewed = sum(1 for image_id in seen if image_id in reviewed_ids)
    # Only clamp to the frames actually present: a review entry can outlive its
    # image if someone deletes frames, and an impossible percentage reads as a
    # bug in the app rather than a stale file.
    reviewed = min(reviewed, total)
    return {
        "total": total,
        "labelled": labelled,
        "boxed": boxed,
        "reviewed": reviewed,
        "unreviewed": max(0, total - reviewed),
        "splits": splits,
        "image_ids": seen,
        "reviewed_ids": reviewed_ids,
    }


def dataset_progress(stats: dict) -> float:
    return round(100.0 * stats["reviewed"] / stats["total"], 1) if stats["total"] else 0.0


def _last_line(text: str) -> str:
    """The most informative line of a failed command: usually the error."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _bounty_progress(bounty: dict, stats: dict) -> dict:
    """How much of one bounty is done.

    Slice-backed bounties are exact. Filter-backed bounties (imported from
    GitHub) can only be estimated, because a bounty there is a target count, not
    a frame list; the estimate is exposed as ``estimated`` so the UI can say so
    rather than implying a precision it does not have.
    """
    slice_ids = bounty.get("slice")
    if slice_ids:
        done = sum(1 for image_id in slice_ids if image_id in stats["reviewed_ids"])
        total = len(slice_ids)
        return {
            "done": done,
            "total": total,
            "percent": round(100.0 * done / total, 1) if total else 0.0,
            "remaining": max(0, total - done),
            "estimated": False,
        }

    target = int(bounty.get("target") or DEFAULT_TARGET)
    # Reviewed-per-split is not tracked, so the estimate uses the whole dataset.
    # That is conservative on a split-scoped bounty: it can only overstate
    # progress if other splits were reviewed, never understate it.
    done = min(stats["reviewed"], target)
    return {
        "done": done,
        "total": target,
        "percent": round(100.0 * done / target, 1) if target else 0.0,
        "remaining": max(0, target - done),
        "estimated": True,
    }


def _bounty_status(bounty: dict, progress: dict) -> str:
    """Status with completion derived from the work, not just the stored flag.

    A bounty whose frames are all reviewed is done even if nobody marked the
    issue: the landing screen should not keep asking for work that is finished.
    """
    if bounty.get("status") == STATUS_DONE:
        return STATUS_DONE
    if progress["total"] and progress["done"] >= progress["total"]:
        return STATUS_DONE
    return bounty.get("status") or STATUS_OPEN


# ------------------------------------------------------------------ datasets
def discover(repo_root: Path) -> list[dict]:
    """Datasets a contributor can review, published first.

    Published datasets live in ``data/datasets/``; scratch ingests live in
    ``.staging/ingested/``. Derived directories such as ``.staging/train`` are
    excluded - they are training output, not something to review.
    """
    repo_root = Path(repo_root)
    found: list[dict] = []
    seen: set[str] = set()

    roots = [
        (repo_root / "data" / "datasets", "published"),
        (repo_root / ".staging" / "ingested", "scratch"),
    ]
    for base, kind in roots:
        if not base.is_dir():
            continue
        for path in sorted(p for p in base.iterdir() if p.is_dir()):
            if path.name.startswith(".") or path.name in seen:
                continue
            if not (path / "images").is_dir() and not (path / "labels").is_dir():
                continue
            seen.add(path.name)
            found.append({"name": path.name, "path": path, "kind": kind})
    return found


# --------------------------------------------------------------------- board
class Board:
    """Local bounty mirror plus dataset catalogue.

    Thread-safe because the HTTP server is threaded: a sync running while the
    landing screen is fetched must not hand back a half-updated board.
    """

    def __init__(self, repo_root: Path, store_path: Path | None = None):
        self.repo_root = Path(repo_root).resolve()
        self.store_path = Path(store_path) if store_path else self.repo_root / ".review" / "board.json"
        self._lock = threading.RLock()
        self._submit_lock = threading.Lock()
        self._data = self._load()

    # ----- persistence ---------------------------------------------------
    def _load(self) -> dict:
        payload: dict = {}
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        bounties = payload.get("bounties")
        payload["bounties"] = bounties if isinstance(bounties, dict) else {}
        payload.setdefault("version", 1)
        payload.setdefault("github", {"synced": 0, "error": "", "user": ""})
        return payload

    def _save(self) -> None:
        self._data["updated"] = time.time()
        _atomic_json(self.store_path, self._data)

    # ----- datasets ------------------------------------------------------
    def datasets(self) -> list[dict]:
        with self._lock:
            out = []
            for entry in discover(self.repo_root):
                stats = _scan_dataset(entry["path"])
                predictions = (entry["path"] / "predictions")
                has_preds = predictions.is_dir() and any(predictions.glob("*.jsonl"))
                out.append({
                    "name": entry["name"],
                    "kind": entry["kind"],
                    "path": str(entry["path"]),
                    "images": stats["total"],
                    "labelled": stats["labelled"],
                    "boxed": stats["boxed"],
                    "reviewed": stats["reviewed"],
                    "unreviewed": stats["unreviewed"],
                    "percent": dataset_progress(stats),
                    "splits": stats["splits"],
                    "has_predictions": has_preds,
                    "has_release": (self.repo_root / "data" / "datasets" / entry["name"] / "manifest.json").exists(),
                })
            # Published first, then most images, so the real datasets lead.
            out.sort(key=lambda d: (d["kind"] != "published", -d["images"], d["name"]))
            return out

    # ----- bounties ------------------------------------------------------
    def _stats_for(self, name: str) -> dict | None:
        for entry in discover(self.repo_root):
            if entry["name"] == name:
                return _scan_dataset(entry["path"])
        return None

    def bounties(self, dataset: str | None = None) -> list[dict]:
        with self._lock:
            stored = list(self._data["bounties"].values())
        stats_cache: dict[str, dict] = {}
        out = []
        for bounty in stored:
            if dataset and bounty.get("dataset") != dataset:
                continue
            name = bounty.get("dataset") or ""
            if name not in stats_cache:
                stats_cache[name] = self._stats_for(name) or {
                    "total": 0, "labelled": 0, "boxed": 0, "reviewed": 0,
                    "unreviewed": 0, "splits": {}, "image_ids": [], "reviewed_ids": set(),
                }
            progress = _bounty_progress(bounty, stats_cache[name])
            out.append({
                **bounty,
                "progress": progress,
                "status": _bounty_status(bounty, progress),
                "missing_dataset": not stats_cache[name]["total"],
            })
        # Order by creation, not status: claiming a bounty must not make its row
        # jump somewhere else in the list under the user's cursor.
        out.sort(key=lambda b: (b.get("dataset", ""), b.get("created", 0), b.get("id", "")))
        return out

    def _next_index(self, dataset: str) -> int:
        used = [
            int(b["index"]) for b in self._data["bounties"].values()
            if b.get("dataset") == dataset and isinstance(b.get("index"), int)
        ]
        return (max(used) + 1) if used else 1

    def seed(
        self,
        dataset: str,
        target: int = DEFAULT_TARGET,
        cap: int = MAX_BOUNTIES,
        force: bool = False,
    ) -> list[dict]:
        """Create slice-backed bounties over a dataset's unreviewed frames.

        Frames already covered by an existing bounty slice are skipped, so
        seeding twice does not create overlapping work.
        """
        stats = self._stats_for(dataset)
        if stats is None:
            raise ValueError(f"unknown dataset: {dataset}")
        if target <= 0:
            raise ValueError("target must be positive")

        with self._lock:
            covered: set[str] = set()
            if not force:
                for bounty in self._data["bounties"].values():
                    if bounty.get("dataset") == dataset:
                        covered.update(bounty.get("slice") or [])

            pending = [i for i in stats["image_ids"] if i not in stats["reviewed_ids"] and i not in covered]
            if not pending:
                return []

            created = []
            chunks = [pending[i:i + target] for i in range(0, len(pending), target)][:cap]
            total = len(chunks)
            for i, chunk in enumerate(chunks, start=1):
                index = self._next_index(dataset)
                bounty = {
                    "id": f"{dataset}#{index}",
                    "dataset": dataset,
                    "index": index,
                    "title": f"{dataset} · Part {i} of {total}",
                    "target": len(chunk),
                    "slice": chunk,
                    "split": "all",
                    "points": len(chunk),
                    "status": STATUS_OPEN,
                    "claimed_by": "",
                    "claimed_at": 0,
                    "github_issue": 0,
                    "github_url": "",
                    "source": "local",
                    "created": time.time(),
                }
                self._data["bounties"][bounty["id"]] = bounty
                created.append(bounty)
            if created:
                self._save()
            return created

    def claim(self, bounty_id: str, by: str = "") -> dict:
        with self._lock:
            bounty = self._data["bounties"].get(bounty_id)
            if bounty is None:
                raise ValueError(f"unknown bounty: {bounty_id}")
            if bounty.get("status") == STATUS_DONE:
                raise ValueError("that bounty is already complete")
            bounty["status"] = STATUS_CLAIMED
            bounty["claimed_by"] = by or bounty.get("claimed_by") or "you"
            bounty["claimed_at"] = time.time()
            bounty["claim_pending_github"] = True
            self._save()
            return dict(bounty)

    def unclaim(self, bounty_id: str) -> dict:
        with self._lock:
            bounty = self._data["bounties"].get(bounty_id)
            if bounty is None:
                raise ValueError(f"unknown bounty: {bounty_id}")
            bounty["status"] = STATUS_OPEN
            bounty["claimed_by"] = ""
            bounty["claimed_at"] = 0
            bounty["claim_pending_github"] = False
            self._save()
            return dict(bounty)

    def link_github(self, bounty_id: str, issue: int, url: str) -> None:
        with self._lock:
            bounty = self._data["bounties"].get(bounty_id)
            if bounty is None:
                return
            bounty["github_issue"] = int(issue)
            bounty["github_url"] = url
            self._save()

    def _link_candidate(self, dataset: str, ordinal: int) -> dict | None:
        """A local bounty that an issue is about, if it has no issue yet.

        Must be called with the lock held (sync holds it; the lock is reentrant).
        """
        if not dataset or not ordinal:
            return None
        exact = self._data["bounties"].get(f"{dataset}#{ordinal}")
        if exact and not exact.get("github_issue"):
            return exact
        for bounty in self._data["bounties"].values():
            if (bounty.get("dataset") == dataset
                    and bounty.get("index") == ordinal
                    and not bounty.get("github_issue")):
                return bounty
        return None

    def gh_connected(self) -> bool:
        """Whether the last sync found a usable GitHub identity."""
        with self._lock:
            github = dict(self._data.get("github") or {})
        return bool(github.get("user")) and not github.get("error")

    # ----- GitHub sync ---------------------------------------------------
    def _gh(self, *args: str, timeout: float = 20.0) -> str:
        res = subprocess.run(
            ["gh", *args], cwd=self.repo_root, capture_output=True, text=True,
            timeout=timeout,
        )
        if res.returncode != 0:
            raise RuntimeError((res.stderr or res.stdout or "gh failed").strip().splitlines()[0])
        return (res.stdout or "").strip()

    def gh_user(self) -> str:
        try:
            return self._gh("api", "user", "--jq", ".login", timeout=10.0)
        except Exception:  # noqa: BLE001
            return ""

    def sync(self) -> dict:
        """Pull issues into the local mirror.

        Failures are recorded on the board instead of raised: the landing screen
        is useful without GitHub, and the UI reads ``github.error`` to explain
        why a sync did not happen.
        """
        try:
            raw = self._gh(
                "issue", "list", "--label", BOUNTY_LABEL, "--state", "all",
                "--json", "number,title,body,labels,url,comments", "--limit", "200",
                timeout=25.0,
            )
            issues = json.loads(raw or "[]")
        except Exception as exc:  # noqa: BLE001
            failure = gh_failure(exc)
            with self._lock:
                self._data["github"] = {
                    **(self._data.get("github") or {}),
                    "synced": time.time(),
                    **failure,
                }
                self._save()
            return {"ok": False, "imported": 0, "updated": 0, **failure}

        imported = 0
        updated = 0
        linked = 0
        with self._lock:
            by_issue = {
                b.get("github_issue"): b for b in self._data["bounties"].values()
                if b.get("github_issue") and not b["id"].startswith("gh#")
            }
            for issue in issues:
                meta = META_RE.search(issue.get("body") or "")
                if not meta:
                    continue
                try:
                    info = json.loads(meta.group(1))
                except ValueError:
                    continue
                labels = {l.get("name") for l in issue.get("labels") or []}
                claimers = [
                    name
                    for c in issue.get("comments") or []
                    for name in CLAIM_RE.findall(c.get("body") or "")
                ]
                number = int(issue.get("number") or 0)
                status = STATUS_OPEN
                if LABEL_DONE in labels:
                    status = STATUS_DONE
                elif claimers or LABEL_CLAIMED in labels:
                    status = STATUS_CLAIMED

                dataset = str(info.get("dataset") or "")
                # Identity beats issue number. A local bounty seeded here has no
                # issue number yet, while a `gh#N` row imported by an earlier
                # sync already claims this one. Matching by number first would
                # keep the shadow row and orphan the row that owns the slice.
                bounty = self._link_candidate(dataset, _issue_ordinal(issue.get("title")))
                if bounty is not None:
                    linked += 1
                else:
                    bounty = by_issue.get(number)
                if bounty is None:
                    bounty_id = f"gh#{number}"
                    bounty = self._data["bounties"].setdefault(bounty_id, {
                        "id": bounty_id,
                        "dataset": dataset,
                        "index": 0,
                        "slice": None,
                        "source": "github",
                        "created": time.time(),
                    })
                    imported += 1
                else:
                    updated += 1

                bounty.update({
                    "title": issue.get("title") or bounty.get("title"),
                    "target": int(info.get("target") or bounty.get("target") or DEFAULT_TARGET),
                    "split": info.get("split") or bounty.get("split") or "all",
                    "filter": info.get("filter") or bounty.get("filter") or {},
                    "points": int(info.get("points") or bounty.get("points") or 0),
                    "github_issue": number,
                    "github_url": issue.get("url") or bounty.get("github_url") or "",
                    "claimed_by": (claimers[0] if claimers else bounty.get("claimed_by") or ""),
                })
                if status == STATUS_DONE:
                    bounty["status"] = STATUS_DONE
                    bounty["claim_pending_github"] = False
                elif status == STATUS_CLAIMED:
                    bounty["status"] = STATUS_CLAIMED
                    bounty["claim_pending_github"] = False
                elif not bounty.get("claim_pending_github"):
                    # Remote says open and nothing is queued locally. When a
                    # claim *is* queued, an open issue is the state we are about
                    # to change — overwriting it would silently drop the claim.
                    bounty["status"] = STATUS_OPEN

            # `gh#N` rows whose issue now belongs to a real bounty are stale
            # duplicates of the same work, minus the slice and the claim.
            owned = {
                b["github_issue"] for b in self._data["bounties"].values()
                if b.get("github_issue") and not b["id"].startswith("gh#")
            }
            for bounty_id in [
                bid for bid, b in self._data["bounties"].items()
                if bid.startswith("gh#") and b.get("github_issue") in owned
            ]:
                del self._data["bounties"][bounty_id]

            self._data["github"] = {
                "synced": time.time(),
                "error": "",
                "kind": "",
                "hint": "",
                "user": (self._data.get("github") or {}).get("user", ""),
            }
            self._save()

        user = self.gh_user()
        if user:
            with self._lock:
                self._data["github"]["user"] = user
                self._save()
        # Claims made while GitHub was down are published now that we can reach
        # it, so "claimed locally" does not mean "claimed only locally" forever.
        flushed = self.flush_pending()
        return {"ok": True, "error": "", "kind": "", "hint": "",
                "imported": imported, "updated": updated, "linked": linked,
                "flushed": flushed}

    def _remote_issues(self) -> list[dict]:
        raw = self._gh(
            "issue", "list", "--label", BOUNTY_LABEL, "--state", "all",
            "--json", "number,title,body,url", "--limit", "200", timeout=25.0,
        )
        return json.loads(raw or "[]")

    def find_issue(self, bounty: dict) -> dict | None:
        """The GitHub issue describing a bounty, matched by dataset + ordinal."""
        dataset = str(bounty.get("dataset") or "")
        ordinal = int(bounty.get("index") or 0)
        if not dataset or not ordinal:
            return None
        for issue in self._remote_issues():
            meta = META_RE.search(issue.get("body") or "")
            if not meta:
                continue
            try:
                info = json.loads(meta.group(1))
            except ValueError:
                continue
            if str(info.get("dataset") or "") != dataset:
                continue
            if _issue_ordinal(issue.get("title")) == ordinal:
                return issue
        return None

    def _create_issue(self, bounty: dict) -> dict | None:
        """Open the GitHub issue for a local bounty.

        Mirrors ``scripts/bounty.py create`` closely enough that the CLI can
        claim, list and submit against an issue the app opened.
        """
        dataset = str(bounty.get("dataset") or "")
        index = int(bounty.get("index") or 0)
        if not dataset or not index:
            return None
        target = int(bounty.get("target") or 0)
        points = int(bounty.get("points") or target)
        peers = [
            b for b in self._data["bounties"].values()
            if b.get("dataset") == dataset
        ]
        meta = {
            "dataset": dataset,
            "split": bounty.get("split") or "all",
            "target": target,
            "filter": bounty.get("filter") or {},
            "points": points,
        }
        title = f"Bounty: {dataset} — {target} frames [{index}/{max(len(peers), index)}]"
        body = "\n".join([
            f"## {title}",
            "",
            f"Review **{target}** frames from `{dataset}`.",
            "",
            "**How to claim**",
            "",
            "Press Claim on the bounty board, or comment `claim by @your-handle` here.",
            "No repository role is needed. The board posts the claim for you, and a",
            "workflow accepts it unless the account looks disposable.",
            "",
            "**What to do**",
            "",
            "- Judge each frame: `g` good, `b` needs fixing (+ a reason), `n` not gameplay.",
            "- Adopt the model's missing boxes with `a` where it is right.",
            "- Push your labels and open a PR that says `Closes #<this issue>`.",
            "",
            f"**Reward:** {points} points",
            "",
            dump_meta(meta),
        ])
        try:
            # Labels must exist before an issue can carry them. Idempotent.
            self._gh("label", "create", BOUNTY_LABEL, "--color", "4a9eff",
                     "--description", "A review task worth points", "--force",
                     timeout=20.0)
        except Exception:  # noqa: BLE001 - existing label is the normal case
            pass
        try:
            url = self._gh("issue", "create", "--title", title, "--body", body,
                           "--label", BOUNTY_LABEL, timeout=30.0)
        except Exception:  # noqa: BLE001
            return None
        number = url.rstrip("/").rsplit("/", 1)[-1]
        # Deliberately *not* locked. A locked issue only accepts comments from
        # people with write access, which would force every labeler to be a
        # collaborator who can push code to main. Spam is filtered by
        # .github/workflows/claim-guard.yml instead.
        return {"number": int(number), "url": url}

    def ensure_linked(self, bounty_id: str, create: bool = True) -> dict | None:
        """Point a local bounty at its issue, opening one when none exists.

        Without this a locally seeded bounty never has an issue number, so its
        claim can only ever be local - which is exactly how a claim ends up
        looking claimed here and untouched on GitHub.
        """
        with self._lock:
            bounty = self._data["bounties"].get(bounty_id)
            if bounty is None:
                raise ValueError(f"unknown bounty: {bounty_id}")
            if bounty.get("github_issue"):
                return {
                    "number": int(bounty["github_issue"]),
                    "url": bounty.get("github_url") or "",
                }
            snapshot = dict(bounty)
        issue = self.find_issue(snapshot)
        if issue is None and create:
            issue = self._create_issue(snapshot)
        if issue is None:
            return None
        self.link_github(bounty_id, int(issue["number"]), issue.get("url") or "")
        return issue

    def post_claim(self, bounty_id: str, create_issue: bool = True) -> dict:
        """Post the local claim to GitHub, linking or opening an issue first.

        Local-first means the claim already counts before this runs. If it fails
        (no auth, locked issue, offline) the claim stays local and
        ``claim_pending_github`` keeps the UI honest about it.
        """
        with self._lock:
            bounty = self._data["bounties"].get(bounty_id)
            if bounty is None:
                raise ValueError(f"unknown bounty: {bounty_id}")

        try:
            issue = self.ensure_linked(bounty_id, create=create_issue)
        except Exception as exc:  # noqa: BLE001
            failure = gh_failure(exc)
            return {"ok": False, "posted": False, "linked": False,
                    "reason": failure["error"], **failure}
        if not issue:
            return {"ok": True, "posted": False, "linked": False,
                    "reason": "no GitHub issue could be linked"}

        number = int(issue["number"])
        with self._lock:
            bounty = self._data["bounties"].get(bounty_id) or {}
            user = self.gh_user() or bounty.get("claimed_by") or ""
        try:
            self._gh("issue", "comment", str(number), "--body", f"claim by @{user}", timeout=20.0)
        except Exception as exc:  # noqa: BLE001
            failure = gh_failure(exc)
            return {"ok": False, "posted": False, "linked": True, "issue": number,
                    "reason": failure["error"], **failure}

        try:
            self._gh("issue", "edit", str(number), "--add-label", LABEL_CLAIMED, timeout=20.0)
        except Exception:  # noqa: BLE001 - the comment alone already marks it claimed
            pass

        with self._lock:
            bounty = self._data["bounties"].get(bounty_id)
            if bounty:
                bounty["claim_pending_github"] = False
                bounty["github_issue"] = number
                bounty["github_url"] = issue.get("url") or bounty.get("github_url") or ""
                self._save()
        return {"ok": True, "posted": True, "linked": True, "issue": number, "reason": ""}

    # ----- submitting finished work ---------------------------------------
    def submit_one(self, bounty_id: str) -> dict:
        """Open a PR for one completed bounty.

        Only the command line knows how to move git branches and open PRs, so
        this shells out to ``scripts/bounty.py submit --bounty`` rather than
        reimplementing it. The scoped submit stages just this bounty's slice,
        which is what keeps one-bounty-one-PR true when several are in flight.
        """
        # bounties() is where progress and status are computed, so the raw
        # record is not enough: it has no "progress" key at all, and a 0/0 read
        # there made every bounty look incomplete.
        bounty = next((b for b in self.bounties() if b.get("id") == bounty_id), None)
        if not bounty:
            return {"id": bounty_id, "ok": False, "reason": "unknown bounty"}
        bounty = dict(bounty)

        # Submitted is terminal. Re-opening the same PR would ship the same
        # branch a second time, and the labels are frozen anyway, so there is
        # nothing a resubmit could add.
        if bounty.get("pr_url") or bounty.get("submitted_at"):
            return {"id": bounty_id, "ok": False,
                    "reason": "already submitted - its labels are locked"}

        progress = bounty.get("progress") or {}
        if bounty.get("status") != "done":
            return {"id": bounty_id, "ok": False,
                    "reason": f"not complete ({progress.get('done', 0)}/"
                              f"{progress.get('total', 0)} frames)"}
        owner = str(bounty.get("claimed_by") or "")
        me = str(self.gh_user() or "")
        if not owner:
            # A bounty can reach 100% because other people reviewed its frames.
            # Submitting it would open a PR whose author did none of the work.
            return {"id": bounty_id, "ok": False,
                    "reason": "nobody claimed it, so there is no work to submit"}
        if me and owner.lower() != me.lower():
            return {"id": bounty_id, "ok": False,
                    "reason": f"claimed by {owner}, not you"}
        number = int(bounty.get("github_issue") or 0)
        if not number:
            return {"id": bounty_id, "ok": False,
                    "reason": "no GitHub issue yet — post it from the bounty first"}
        if not self.gh_connected():
            return {"id": bounty_id, "ok": False,
                    "reason": "GitHub is not connected, so a PR cannot be opened"}

        cmd = [sys.executable, str(self.repo_root / "scripts" / "bounty.py"),
               "submit", "--bounty", bounty_id, "--issue", str(number), "--push"]
        try:
            res = subprocess.run(cmd, cwd=self.repo_root, capture_output=True,
                                 text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return {"id": bounty_id, "ok": False,
                    "reason": "submit timed out after 5 minutes"}
        except OSError as exc:
            return {"id": bounty_id, "ok": False, "reason": f"could not run submit: {exc}"}

        output = (res.stdout or "") + (res.stderr or "")
        url_match = re.search(r"Opened PR: (\S+)", output)
        if res.returncode != 0:
            return {"id": bounty_id, "ok": False,
                    "reason": _last_line(output) or f"submit failed ({res.returncode})",
                    "output": output[-2000:]}

        url = url_match.group(1) if url_match else ""
        with self._lock:
            stored = self._data["bounties"].get(bounty_id)
            if stored:
                stored["pr_url"] = url
                stored["submitted_at"] = int(time.time())
                self._save()
        return {"id": bounty_id, "ok": True, "pr_url": url, "output": output[-2000:],
                "reason": ""}

    def submit(self, bounty_ids: list[str]) -> dict:
        """Submit completed bounties one at a time.

        Serialised because each submit checks out a branch; two concurrent runs
        would fight over the same working tree.
        """
        with self._submit_lock:
            results = [self.submit_one(bid) for bid in bounty_ids]
        return {"ok": all(r["ok"] for r in results) if results else True,
                "results": results}

    def flush_pending(self) -> dict:
        """Publish claims that were made while GitHub was unreachable.

        The board promises a queued claim posts on the next sync; this is the
        code that keeps that promise. It never opens new issues - a sync should
        not create work as a side effect.
        """
        with self._lock:
            pending = [
                bid for bid, b in self._data["bounties"].items()
                if b.get("claim_pending_github")
            ]
        posted = 0
        failed = 0
        for bounty_id in pending:
            try:
                result = self.post_claim(bounty_id, create_issue=False)
            except Exception:  # noqa: BLE001
                failed += 1
                continue
            if result.get("posted"):
                posted += 1
            else:
                failed += 1
        return {"posted": posted, "failed": failed}

    # ----- serialisation -------------------------------------------------
    def snapshot(self, dataset: str | None = None) -> dict:
        with self._lock:
            github = dict(self._data.get("github") or {})
        return {
            "datasets": self.datasets(),
            "bounties": self.bounties(dataset),
            "github": github,
        }
