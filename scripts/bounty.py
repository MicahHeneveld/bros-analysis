#!/usr/bin/env python3
"""Bounty board for the SSBU label review app.

Bounties are GitHub Issues. A bounty is a *review target*, not a fixed range of
frames: a dataset, a filter, and a number of frames to judge. That matters,
because the app already sorts unreviewed frames first, so two people claiming
overlapping bounties naturally work on different frames without any server-side
locking. Claims are therefore advisory, not exclusive -- see docs/contributing.md.

Issue bodies carry machine-readable metadata in an HTML comment, which GitHub
does not render:

    <!-- bounty: {"dataset": "ssbu_players", "split": "train", "target": 250,
                  "filter": {"kind": "unlabelled_player"}, "difficulty": 1.31,
                  "points": 328} -->

Points are `target * difficulty`, fixed at creation time so the reward cannot be
inflated later. Difficulty is the model's own uncertainty on that slice: frames
the model is unsure about are the frames worth a human's attention.

Usage:
    python scripts/bounty.py doctor
    python scripts/bounty.py create --dataset data/datasets/ssbu_players --target 250
    python scripts/bounty.py list
    python scripts/bounty.py claim 12
    python scripts/bounty.py start 12
    python scripts/bounty.py submit
    python scripts/bounty.py leaderboard
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META_RE = re.compile(r"<!--\s*bounty:\s*(\{.*?\})\s*-->", re.S)
# A claim is exactly what `claim` posts: "claim by @user". Matching a bare
# substring counted unrelated comments that merely used the word "claim"
# (a 22KB spam comment did), which made untouched bounties look taken.
CLAIM_RE = re.compile(r"^[ \t]*claim by @([A-Za-z0-9-]+)[ \t]*$", re.I | re.M)

LABEL_BOUNTY = "bounty"
LABEL_CLAIMED = "bounty:claimed"
LABEL_DONE = "bounty:done"


# --------------------------------------------------------------------- shell
def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=capture,
    )


def gh(*args: str, check: bool = True) -> str:
    """Run a gh subcommand and return stdout."""
    res = run(["gh", *args], check=check, capture=True)
    return res.stdout.strip() if res.stdout else ""


def fail(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


# ------------------------------------------------------------------ metadata
def parse_meta(body: str) -> dict | None:
    m = META_RE.search(body or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def dump_meta(meta: dict) -> str:
    return "<!-- bounty: " + json.dumps(meta, sort_keys=True) + " -->"


def points_for(target: int, difficulty: float) -> int:
    """Round outward so a bounty is never worth less than advertised."""
    return int(target * difficulty + 0.999)


# ---------------------------------------------------------------- difficulty
def slice_difficulty(predictions_jsonl: Path | None, limit: int) -> float:
    """Estimate how hard a slice is from the model's confidence.

    A frame the model is confident about is quick to confirm; a frame where it
    is unsure or detects nothing is where a human earns their keep. The returned
    multiplier sits in [1.0, 2.0].
    """
    if not predictions_jsonl or not predictions_jsonl.exists():
        return 1.0

    scores: list[float] = []
    with predictions_jsonl.open() as fh:
        for line in fh:
            if len(scores) >= limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            boxes = rec.get("predictions_clean") or rec.get("predictions") or []
            if not boxes:
                # Nothing detected at all: probably background, but it needs a
                # human to confirm that, so treat it as uncertain.
                scores.append(0.75)
                continue
            confs = [float(b.get("conf", 0.0)) for b in boxes]
            avg = sum(confs) / len(confs)
            # One solid detection is an easy confirm; several weak ones are not.
            scores.append(min(1.0, (1.0 - avg) + 0.15 * max(0, len(boxes) - 2)))

    if not scores:
        return 1.0
    return round(1.0 + (sum(scores) / len(scores)), 2)


def find_predictions(dataset: Path) -> Path | None:
    for cand in (dataset / "predictions", ROOT / ".staging" / "predictions"):
        if cand.is_dir():
            files = sorted(cand.glob("*.jsonl"))
            if files:
                return files[0]
    return None


def release_assets() -> list[dict]:
    """Images are published as release assets, so pulling them needs no auth.

    Tag convention: `images-<dataset>`. Anything else in the release list is
    ignored, so a release can also hold model weights or notes without confusing
    the fetch.

    Note: `gh release list` cannot return assets (only metadata), so each
    candidate tag needs a second `gh release view` call. There are normally only
    a handful, and it avoids depending on the REST payload shape.
    """
    try:
        tags_raw = gh("release", "list", "--json", "tagName", "--limit", "100")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    try:
        tags = [r.get("tagName") or "" for r in json.loads(tags_raw or "[]")]
    except json.JSONDecodeError:
        return []

    out = []
    for tag in tags:
        if not tag.startswith("images-"):
            continue
        try:
            view = json.loads(gh("release", "view", tag, "--json", "assets") or "{}")
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        for asset in view.get("assets") or []:
            out.append({
                "dataset": tag[len("images-"):],
                "tag": tag,
                "name": asset.get("name", ""),
                "size": asset.get("size", 0),
                "url": asset.get("url", ""),
            })
    return out


def download_images(dataset_name: str, dest: Path, force: bool = False) -> bool:
    """Fetch and extract the image bundle for one dataset.

    Extraction is deliberately staged: the archive is downloaded to a temp file,
    checked to be a real archive, and only then unpacked. A truncated download
    that half-fills the images directory is worse than no images at all, because
    the app would silently show a partial dataset.
    """
    matches = [a for a in release_assets() if a["dataset"] == dataset_name]
    if not matches:
        print(f"  No release tagged 'images-{dataset_name}'.")
        print(f"  Publish one with:  gh release create images-{dataset_name} <bundle>.zip")
        return False

    asset = matches[0]
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()) and not force:
        print(f"  {dest} is not empty. Pass --force to overwrite.")
        return False

    size_mb = asset["size"] / (1024 * 1024)
    print(f"  Downloading {asset['name']} ({size_mb:.1f} MB) from {asset['tag']}…")

    tmpdir = Path(tempfile.mkdtemp(prefix="ssbu-images-"))
    try:
        # --dir, not --output: gh refuses to write over a path that already
        # exists, and --output pre-creates it.
        subprocess.run(
            ["gh", "release", "download", asset["tag"],
             "--pattern", asset["name"], "--dir", str(tmpdir)],
            cwd=ROOT, check=True,
        )
        archive = tmpdir / asset["name"]
        if not archive.exists():
            print(f"  gh reported success but {asset['name']} is not there.")
            return False

        # Verify against the size the API reported before trusting the file.
        actual = archive.stat().st_size
        if asset["size"] and actual != asset["size"]:
            print(f"  Download is {actual} bytes, expected {asset['size']} — aborting.")
            return False

        extract_to = tmpdir / "x"
        extract_to.mkdir()
        # unpack_archive returns None, so this has to be exception-based rather
        # than a truthiness check.
        try:
            shutil.unpack_archive(str(archive), str(extract_to))
        except (shutil.ReadError, OSError) as exc:
            print(f"  Could not extract {asset['name']}: {exc}")
            return False

        # Bundles are zipped differently depending on how they were made: a bare
        # `images/` folder, or a wrapping `dataset/images/`. Find the directory
        # that actually holds the images and flatten *its contents* into dest, so
        # the result is always <dataset>/images/<file>.
        src = _images_root(extract_to)
        if src is None:
            print("  Extract succeeded but no images were found inside the bundle.")
            return False

        n = _copy_tree(src, dest)
        print(f"  Extracted {n} image(s) into {dest}")
        return n > 0
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"  Download failed: {exc}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _direct_images(root: Path) -> int:
    """Images sitting directly in `root`, not in a subdirectory."""
    try:
        return sum(1 for p in root.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    except OSError:
        return 0


def _images_root(root: Path) -> Path | None:
    """Find the directory whose *contents* are the dataset's image tree.

    A directory literally named `images` is authoritative even when it holds no
    files directly, because YOLO datasets put splits underneath it
    (`images/train/...`). Treating that as a wrapper to be stripped would
    flatten the splits and leave the reviewer's dataset shaped differently from
    the one the labels were made against.

    Only when no `images` directory exists do we fall back to guessing from
    where the files actually are.
    """
    named = [p for p in root.rglob("*") if p.is_dir() and p.name == "images"]
    if named:
        # Shallowest wins: the dataset's `images/` over a stray `images/cache`.
        return sorted(named, key=lambda p: len(p.parts))[0]

    candidates = [p for p in root.rglob("*") if p.is_dir() and _direct_images(p)]
    if candidates:
        return sorted(candidates, key=lambda p: -len(p.parts))[0]
    return None


def _copy_tree(src: Path, dest: Path) -> int:
    n = 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        n += 1
    return n


def cmd_fetch(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset).resolve()
    ok = download_images(dataset.name, dataset / "images", force=args.force)
    return 0 if ok else 1


# -------------------------------------------------------------------- doctor
def cmd_doctor(args: argparse.Namespace) -> int:
    print("Checking your setup\n")
    ok = True

    def check(name: str, good: bool, detail: str, fix: str = "") -> None:
        nonlocal ok
        mark = "ok  " if good else "MISS"
        print(f"  [{mark}] {name}: {detail}")
        if not good:
            ok = False
            if fix:
                print(f"         -> {fix}")

    check("git", shutil.which("git") is not None, "found" if shutil.which("git") else "not installed",
          "install git")
    check("repo", (ROOT / ".git").exists(), str(ROOT))

    try:
        user = gh("api", "user", "--jq", ".login", check=True)
        check("gh auth", True, f"signed in as {user}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        check("gh auth", False, "not signed in or gh missing", "gh auth login")

    dataset = (ROOT / args.dataset).resolve()
    check("dataset", (dataset / "data.yaml").exists() or (dataset / "labels").is_dir(),
          str(dataset), "pass --dataset, or stage one with scripts/stage_dataset.py")

    images = dataset / "images"
    n_images = len(list(images.rglob("*.jpg"))) + len(list(images.rglob("*.png"))) if images.is_dir() else 0
    check("images", n_images > 0, f"{n_images} files",
          f"run  python scripts/bounty.py fetch --dataset {args.dataset}")

    # Tell the contributor where the images come from before they go hunting.
    assets = [a for a in release_assets() if a["dataset"] == dataset.name]
    if assets:
        size_mb = assets[0]["size"] / (1024 * 1024)
        print(f"         available: {assets[0]['tag']} / {assets[0]['name']} ({size_mb:.0f} MB)")
    elif n_images == 0:
        print(f"         no release tagged 'images-{dataset.name}' exists yet")

    check("python", sys.version_info >= (3, 9), sys.version.split()[0])
    check("docker", shutil.which("docker") is not None or True, "optional (local container run)")

    print()
    if ok:
        print("Ready. Claim a bounty with:  python scripts/bounty.py list")
    else:
        print("Fix the MISS lines above, then re-run  python scripts/bounty.py doctor")
    return 0 if ok else 1



# -------------------------------------------------------------------- create
def labelled_frames(dataset: Path) -> int:
    """How many label files actually contain boxes.

    Zero means the dataset is unlabelled, which changes what a bounty *is*: the
    contributor is creating labels rather than checking existing ones. Saying
    "review these frames" on an empty dataset would be wrong about the work.
    """
    labels = dataset / "labels"
    if not labels.is_dir():
        return 0
    return sum(1 for p in labels.rglob("*.txt") if p.stat().st_size > 0)


def cmd_create(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        fail(f"dataset not found: {dataset}")

    name = dataset.name
    preds = find_predictions(dataset)
    difficulty = slice_difficulty(preds, args.target)
    pts = points_for(args.target, difficulty)
    n_labelled = labelled_frames(dataset)
    bootstrap = n_labelled == 0

    meta = {
        "dataset": name,
        "split": args.split,
        "target": args.target,
        "filter": {k: v for k, v in {"kind": args.kind, "flag": args.flag}.items() if v},
        "difficulty": difficulty,
        "points": pts,
        "bootstrap": bootstrap,
    }
    title = f"Bounty: {name} — {args.target} frames"
    if bootstrap:
        title += " (label from scratch)"
    elif args.kind:
        title += f" ({args.kind})"
    # Several bounties over the same dataset need distinct titles, or the
    # duplicate guard refuses the second one.
    if args.count > 1:
        title += f" [{args.batch}/{args.count}]"

    if bootstrap:
        goal = (f"Create labels for **{args.target}** frames from `{name}`. "
                "This dataset has no labels yet, so you are the source of truth — "
                "the model's boxes are a starting point, not an answer.")
        tasks = [
            "- Open each frame. The dashed boxes are the model's guesses.",
            "- Press `a` to adopt them all, then fix what is wrong; or press `d` to draw by hand.",
            "- Every player gets exactly one tight box. Press `g` when the frame is correct.",
            "- Press `n` on menus, results screens and other non-gameplay frames.",
        ]
    else:
        goal = (f"Review **{args.target}** frames from `{name}`"
                + (f", filtered to `{args.kind}`." if args.kind else "."))
        tasks = [
            "- Judge each frame: `g` good, `b` needs fixing (+ a reason), `n` not gameplay.",
            "- Adopt the model's missing boxes with `a` where it is right.",
        ]

    body = "\n".join([
        f"## {title}",
        "",
        goal,
        "",
        "**How to claim**",
        "",
        f"Comment `claim` on this issue. Then run `python scripts/bounty.py start {args.issue_hint or '<number>'}`.",
        "",
        "**What to do**",
        "",
        *tasks,
        "- Push your labels and open a PR that says `Closes #<this issue>`.",
        "",
        f"**Reward:** {pts} points "
        f"({args.target} frames × {difficulty} difficulty)",
        "",
        dump_meta(meta),
    ])

    if args.dry_run:
        print(body)
        return 0

    # Labels have to exist before an issue can carry them. Cheap and idempotent.
    for name, colour, desc in (
        (LABEL_BOUNTY, "4a9eff", "A review task worth points"),
        (LABEL_CLAIMED, "e0a437", "Someone has claimed this bounty"),
        (LABEL_DONE, "35c07a", "Bounty completed and merged"),
    ):
        run(["gh", "label", "create", name, "--color", colour,
             "--description", desc, "--force"], check=False)

    # Ensure we are not opening a duplicate of an issue that is still open.
    existing = json.loads(gh("issue", "list", "--label", LABEL_BOUNTY,
                             "--state", "open", "--json", "number,title",
                             "--limit", "200") or "[]")
    if any(i.get("title") == title for i in existing):
        print(f"An open bounty with this title already exists: {title}")
        return 0

    out = gh("issue", "create", "--title", title, "--body", body,
             "--label", LABEL_BOUNTY)
    print(f"Opened bounty: {out}")
    return 0


# ---------------------------------------------------------------------- list
def cmd_list(args: argparse.Namespace) -> int:
    raw = gh("issue", "list", "--label", LABEL_BOUNTY, "--state", "open",
             "--json", "number,title,body,labels,assignees,comments", "--limit", "100")
    issues = json.loads(raw or "[]")
    if not issues:
        print("No open bounties. Create some with:  python scripts/bounty.py create --dataset ...")
        return 0

    print(f"{'#':>5}  {'points':>6}  {'claims':>6}  title")
    print("-" * 72)
    total = 0
    for issue in issues:
        meta = parse_meta(issue.get("body") or "") or {}
        labels = {l["name"] for l in issue.get("labels", [])}
        if LABEL_DONE in labels:
            continue
        pts = meta.get("points", 0)
        total += pts
        claims = sum(len(CLAIM_RE.findall(c.get("body") or ""))
                     for c in issue.get("comments", []))
        state = " (claimed)" if claims else ""
        print(f"{issue['number']:>5}  {pts:>6}  {claims:>6}  {issue['title']}{state}")
    print("-" * 72)
    print(f"{total} points available")
    return 0


# --------------------------------------------------------------------- claim
def load_bounty(number: int) -> tuple[dict, dict]:
    raw = gh("issue", "view", str(number), "--json", "number,title,body,labels")
    issue = json.loads(raw)
    meta = parse_meta(issue.get("body") or "")
    if not meta:
        fail(f"issue #{number} has no bounty metadata — is it a bounty?")
    return issue, meta


def cmd_claim(args: argparse.Namespace) -> int:
    issue, meta = load_bounty(args.issue)
    me = gh("api", "user", "--jq", ".login")

    existing = gh("issue", "view", str(args.issue), "--json", "comments",
                  "--jq", "[.comments[].body] | join(\"\\n\")") or ""
    claimers = {name.lower() for name in CLAIM_RE.findall(existing)}
    if me.lower() in claimers:
        print(f"You already claimed #{args.issue}.")
    elif claimers and not args.force:
        print(f"#{args.issue} is already claimed by someone else. Claims are advisory, so")
        print("you can still work on it — but coordinate on the issue first, or pass --force.")
        return 1
    else:
        gh("issue", "comment", str(args.issue), "--body", f"claim by @{me}")
        print(f"Claimed #{args.issue}.")

    try:
        gh("issue", "edit", str(args.issue), "--add-label", LABEL_CLAIMED)
    except subprocess.CalledProcessError:
        pass  # label may not exist yet; not worth failing a claim over

    branch = f"bounty/{args.issue}"
    in_branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture=True).stdout.strip()
    if in_branch != branch:
        dirty = run(["git", "status", "--porcelain"], capture=True).stdout.strip()
        if dirty and not args.force:
            fail("working tree is dirty — commit or stash before claiming (or pass --force)")
        exists = run(["git", "rev-parse", "--verify", branch], check=False, capture=True)
        if exists.returncode == 0:
            run(["git", "checkout", branch])
        else:
            run(["git", "checkout", "-b", branch])
        print(f"On branch {branch}.")

    print(f"\nNext:  python scripts/bounty.py start {args.issue}")
    return 0


# --------------------------------------------------------------------- start
def cmd_start(args: argparse.Namespace) -> int:
    issue, meta = load_bounty(args.issue)
    dataset = (ROOT / args.dataset if args.dataset else ROOT / "data/datasets" / meta["dataset"]).resolve()
    if not dataset.exists():
        fail(f"dataset not found: {dataset}\nRun  python scripts/bounty.py doctor")

    url = f"http://127.0.0.1:{args.port}/?mode=images&bounty={quote(issue['title'])}"
    filt = meta.get("filter") or {}
    if filt.get("kind"):
        url += "&kind=" + quote(filt["kind"])
    if filt.get("flag"):
        url += "&flag=" + quote(filt["flag"])
    if meta.get("split"):
        url += "&split=" + quote(meta["split"])

    if args.no_serve:
        print(url)
        return 0

    cmd = [
        sys.executable, str(ROOT / "app/server.py"),
        "--dataset", str(dataset),
        "--port", str(args.port),
    ]
    preds = dataset / "predictions"
    if preds.is_dir():
        cmd += ["--predictions", str(preds)]

    print(f"Bounty #{args.issue}: {issue['title']}")
    print(f"Worth {meta.get('points', 0)} points.\n")
    print(f"Serving {dataset}")
    print(f"Open: {url}\n")
    print("Ctrl-C to stop. Your reviews save as you go.")
    try:
        subprocess.run(cmd, cwd=ROOT)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def quote(s: str) -> str:
    from urllib.parse import quote as q
    return q(s, safe="")


# -------------------------------------------------------------------- submit
def cmd_submit(args: argparse.Namespace) -> int:
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture=True).stdout.strip()
    if branch in ("main", "master") and not args.force:
        fail("refusing to submit from main — claim a bounty first, which creates a branch")

    number = args.issue
    if not number:
        m = re.search(r"bounty/(\d+)", branch)
        if m:
            number = int(m.group(1))
    if not number:
        fail("cannot tell which issue this branch belongs to; pass --issue N")

    issue, meta = load_bounty(number)

    dirty = run(["git", "status", "--porcelain"], capture=True).stdout.strip()
    if not dirty and not args.force:
        print("No changes to submit.")
        return 0

    # Only labels and review state belong in a bounty PR. Images and staged data
    # are gitignored on purpose, so adding paths broadly is safe but noisy.
    run(["git", "add", "-A", "data/datasets"])
    staged = run(["git", "diff", "--cached", "--name-only"], capture=True).stdout.strip()
    if not staged:
        print("No label changes staged. Did you actually review anything?")
        return 1

    n_files = len(staged.splitlines())
    # Not every staged file is a frame: dataset metadata (data.yaml, manifest,
    # predictions) rides along on the first bounty. Counting those as "frames"
    # would overstate the work in the commit message and the PR body.
    n_labels = sum(1 for p in staged.splitlines()
                   if "/labels/" in p and p.endswith(".txt"))
    pts = meta.get("points", 0)
    run(["git", "commit", "-m",
         f"labels: bounty #{number} ({n_labels} frames)\n\nCloses #{number}"])
    print(f"Committed {n_labels} label file(s)"
          + (f" plus {n_files - n_labels} dataset file(s)" if n_files > n_labels else "")
          + ".")

    if args.push:
        run(["git", "push", "-u", "origin", branch])
        body = "\n".join([
            f"Closes #{number}",
            "",
            f"Bounty: {issue['title']}",
            f"Reviewed {n_labels} frame(s) — worth {pts} points once merged.",
            "",
            "Generated by `scripts/bounty.py submit`.",
        ])
        url = gh("pr", "create", "--title", f"labels: bounty #{number}",
                 "--body", body, "--base", args.base)
        print(f"Opened PR: {url}")
    else:
        print("Committed locally. Re-run with --push to open a PR.")
    return 0


# --------------------------------------------------------------- leaderboard
def cmd_leaderboard(args: argparse.Namespace) -> int:
    raw = gh("pr", "list", "--state", "merged", "--json",
             "number,title,body,author,mergedAt", "--limit", "500")
    prs = json.loads(raw or "[]")

    # Points are read from the bounty issue, so editing an issue later cannot
    # silently change what a merged PR was worth... except that it can. The PR
    # body also records the award at submit time, and that wins.
    awards: dict[str, int] = {}
    per_pr: list[dict] = []
    for pr in prs:
        body = pr.get("body") or ""
        m = re.search(r"worth (\d+) points", body)
        issue_m = re.search(r"Closes #(\d+)", body)
        pts = int(m.group(1)) if m else 0
        if not pts and issue_m:
            try:
                _, meta = load_bounty(int(issue_m.group(1)))
                pts = int(meta.get("points", 0))
            except SystemExit:
                pts = 0
        author = (pr.get("author") or {}).get("login") or "unknown"
        awards[author] = awards.get(author, 0) + pts
        per_pr.append({"pr": pr.get("number"), "author": author, "points": pts,
                       "mergedAt": pr.get("mergedAt")})

    if not awards:
        print("No merged bounty PRs yet.")
        return 0

    print(f"{'points':>7}  contributor")
    print("-" * 40)
    for author, pts in sorted(awards.items(), key=lambda kv: -kv[1]):
        print(f"{pts:>7}  @{author}")
    print("-" * 40)
    print(f"{sum(awards.values())} points across {len(per_pr)} merged PR(s)")

    if args.write:
        out = ROOT / "docs" / "leaderboard.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "totals": awards,
            "prs": per_pr,
        }, indent=2) + "\n")
        print(f"Wrote {out.relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SSBU label bounty board")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check your setup")
    d.add_argument("--dataset", default="data/datasets/ssbu_players")
    d.set_defaults(func=cmd_doctor)

    f = sub.add_parser("fetch", help="download the image bundle for a dataset")
    f.add_argument("--dataset", default="data/datasets/ssbu_players")
    f.add_argument("--force", action="store_true", help="overwrite a non-empty images dir")
    f.set_defaults(func=cmd_fetch)

    c = sub.add_parser("create", help="open bounty issues for a dataset")
    c.add_argument("--dataset", required=True)
    c.add_argument("--target", type=int, default=250, help="frames per bounty")
    c.add_argument("--split", default="train")
    c.add_argument("--kind", default="", help="disagreement filter, e.g. unlabelled_player")
    c.add_argument("--flag", default="", help="label hygiene filter")
    c.add_argument("--issue-hint", default="")
    c.add_argument("--count", type=int, default=1,
                   help="how many bounties to open for this dataset")
    c.add_argument("--batch", type=int, default=1,
                   help="which of --count this invocation is (for resuming)")
    c.add_argument("--dry-run", action="store_true", help="print the issue body only")
    c.set_defaults(func=cmd_create)

    ls = sub.add_parser("list", help="list open bounties")
    ls.set_defaults(func=cmd_list)

    cl = sub.add_parser("claim", help="claim a bounty and branch")
    cl.add_argument("issue", type=int)
    cl.add_argument("--force", action="store_true")
    cl.set_defaults(func=cmd_claim)

    st = sub.add_parser("start", help="launch the review app on a bounty")
    st.add_argument("issue", type=int)
    st.add_argument("--dataset", default="")
    st.add_argument("--port", type=int, default=8770)
    st.add_argument("--no-serve", action="store_true", help="just print the URL")
    st.set_defaults(func=cmd_start)

    sb = sub.add_parser("submit", help="commit labels and open a PR")
    sb.add_argument("--issue", type=int)
    sb.add_argument("--push", action="store_true", help="push and open the PR")
    sb.add_argument("--base", default="main")
    sb.add_argument("--force", action="store_true")
    sb.set_defaults(func=cmd_submit)

    lb = sub.add_parser("leaderboard", help="points from merged PRs")
    lb.add_argument("--write", action="store_true", help="write docs/leaderboard.json")
    lb.set_defaults(func=cmd_leaderboard)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
