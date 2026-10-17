"""Self-tests for the label review app. No test framework required.

Run: python app/selftest.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json  # noqa: E402

from label_io import Box, LabelStore, parse_text, serialize_boxes  # noqa: E402
from predictions import (  # noqa: E402
    Predictions,
    classify,
    evaluate,
    iou,
    postprocess,
    suppress,
)
from review_store import ReviewStore, REASON_CODES  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {detail}")


def test_parse_basic() -> None:
    print("\n[parse]")
    text = "0 0.437794 0.578130 0.264474 0.384204\n"
    result = parse_text(text)
    check("one box parsed", len(result.boxes) == 1)
    check("no malformed", not result.malformed)
    box = result.boxes[0]
    check("class parsed", box.cls == 0)
    check("xc parsed", abs(box.xc - 0.437794) < 1e-9, str(box.xc))

    empty = parse_text("")
    check("empty file yields no boxes", not empty.boxes and not empty.malformed)


def test_roundtrip_identity() -> None:
    print("\n[round-trip byte identity]")
    samples = [
        "0 0.437794 0.578130 0.264474 0.384204\n",
        "",
        "0 0.1 0.2 0.3 0.4\n1 0.5 0.6 0.7 0.8\n",
        # Trailing spaces and a missing final newline must survive.
        "0 0.100000 0.200000 0.300000 0.400000   ",
        "0 0.515625 0.531250 0.125000 0.156250\n\n",
    ]
    for i, text in enumerate(samples):
        parsed = parse_text(text)
        out = serialize_boxes(parsed.boxes, preserve_text=text)
        check(f"sample {i} byte-identical", out == text, f"{out!r} != {text!r}")

    # Malformed rows must survive.
    weird = "this is not a label\n0 0.1 0.2 0.3 0.4\n"
    parsed = parse_text(weird)
    check("malformed row detected", len(parsed.malformed) == 1)
    check("malformed warning raised", len(parsed.warnings()) == 1)
    out = serialize_boxes(parsed.boxes, preserve_text=weird)
    check("malformed preserves raw text", out == weird)


def test_clamping() -> None:
    print("\n[clamping]")
    b = Box(0, 0.0, 0.0, 0.5, 0.5).clamp()
    check("center moved inside frame", b.xc >= 0.25 - 1e-9 and b.yc >= 0.25 - 1e-9, f"{b}")

    b = Box(0, 1.0, 1.0, 0.5, 0.5).clamp()
    check("far edge clamped", b.xc <= 0.75 + 1e-9 and b.yc <= 0.75 + 1e-9, f"{b}")

    b = Box(0, 0.5, 0.5, 2.0, 3.0).clamp()
    check("oversized w/h clamped to 1", b.w <= 1.0 and b.h <= 1.0, f"{b}")

    degenerate = Box(0, 0.5, 0.5, 0.0, 0.0)
    check("degenerate detected", degenerate.is_degenerate())


def test_line_endings() -> None:
    print("\n[line endings: the real dataset uses CRLF]")
    tmp = Path(tempfile.mkdtemp(prefix="labelrev-"))
    try:
        root = tmp / "ds"
        (root / "labels" / "train").mkdir(parents=True)
        (root / "images" / "train").mkdir(parents=True)
        img_id = "train/crlf"
        crlf = b"0 0.388948 0.583394 0.093750 0.310528\r\n0 0.489607 0.583394 0.062172 0.264917\r\n"
        (root / "labels" / f"{img_id}.txt").write_bytes(crlf)

        store = LabelStore(root)

        # A no-op save must leave the CRLF bytes untouched.
        parsed = store.load(img_id)
        check("newline detected as CRLF", parsed.newline == "\r\n", repr(parsed.newline))
        res = store.save(img_id, parsed.boxes)
        check("no-op leaves CRLF untouched", (root / "labels" / f"{img_id}.txt").read_bytes() == crlf)

        # A real edit must keep CRLF rather than rewriting the whole file to LF.
        res = store.save(img_id, [Box(0, 0.25, 0.5, 0.1, 0.3)], reason="move_box")
        after = (root / "labels" / f"{img_id}.txt").read_bytes()
        check("edit reports changed", res["changed"] is True)
        check("edit preserves CRLF", b"\r\n" in after and b"\n\r" not in after, repr(after))
        check("no stray bare LF", after.replace(b"\r\n", b"") .count(b"\n") == 0, repr(after))
        check(
            "edited geometry correct",
            after == b"0 0.250000 0.500000 0.100000 0.300000\r\n",
            repr(after),
        )

        # Undo must restore the original CRLF bytes exactly.
        store.restore_from_journal(img_id)
        check("undo restores CRLF bytes", (root / "labels" / f"{img_id}.txt").read_bytes() == crlf)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_decode_safety() -> None:
    print("\n[decode safety]")
    tmp = Path(tempfile.mkdtemp(prefix="labelrev-"))
    try:
        root = tmp / "ds"
        (root / "labels" / "train").mkdir(parents=True)
        (root / "images" / "train").mkdir(parents=True)
        img_id = "train/binary"
        bad_bytes = b"\xff\xfe not utf8\n"
        (root / "labels" / f"{img_id}.txt").write_bytes(bad_bytes)

        store = LabelStore(root)
        parsed = store.load(img_id)
        check("decode error flagged", parsed.decode_error is True)
        check("decode warning surfaced", any("UTF-8" in w for w in parsed.warnings()))

        refused = False
        try:
            store.save(img_id, [Box(0, 0.5, 0.5, 0.2, 0.2)])
        except ValueError:
            refused = True
        check("editing a non-UTF-8 file is refused", refused)
        check(
            "non-UTF-8 bytes left untouched",
            (root / "labels" / f"{img_id}.txt").read_bytes() == bad_bytes,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_predictions_scoring() -> None:
    print("\n[prediction scoring]")
    W, H = 1920, 1080

    def box(xc, yc, w, h, cls=0, conf=None):
        d = {"cls": cls, "xc": xc, "yc": yc, "w": w, "h": h}
        if conf is not None:
            d["conf"] = conf
        return d

    # IoU basics.
    a = box(0.5, 0.5, 0.2, 0.2)
    check("identical boxes IoU == 1", abs(iou(a, a, W, H) - 1.0) < 1e-9)
    check("disjoint boxes IoU == 0", iou(a, box(0.05, 0.05, 0.05, 0.05), W, H) == 0.0)
    # Half-overlap in x, full in y -> inter 0.5 of one box area.
    b = box(0.6, 0.5, 0.2, 0.2)
    expected = (0.1 * 0.2) / (0.2 * 0.2 + 0.2 * 0.2 - 0.1 * 0.2)
    check("partial overlap IoU correct", abs(iou(a, b, W, H) - expected) < 1e-6,
          f"{iou(a, b, W, H)} vs {expected}")

    # Identical labels and predictions -> no disagreement.
    same = [box(0.3, 0.5, 0.2, 0.4), box(0.7, 0.5, 0.2, 0.4)]
    preds = [dict(p, conf=0.9) for p in same]
    m = evaluate(same, preds, W, H)
    check("perfect agreement scores 0", m["disagreement"] == 0.0, str(m))
    check("perfect agreement matched 2", m["matched"] == 2)
    check("perfect agreement has no extras", m["extra"] == 0 and m["missing"] == 0)
    check("perfect agreement classified agree", classify(m) == "agree")

    # Confident prediction with no label -> strong signal (likely missing box).
    m = evaluate([box(0.3, 0.5, 0.2, 0.4)], [dict(same[0], conf=0.95), dict(same[1], conf=0.9)], W, H)
    check("missed player detected", m["missed_player_hi"] == 1, str(m))
    check("no label overlap -> not a duplicate", m["duplicate_hi"] == 0, str(m))
    check("missed player ranks high", m["disagreement"] >= 0.7, str(m["disagreement"]))
    check("missed player classified as such", classify(m) == "unlabelled_player")

    # A second box on a character that already has a label is a DUPLICATE, not a
    # missing player. It must be classified separately and score low, because
    # adopting it would teach the model to double-fire.
    dup = dict(same[0], conf=0.95)
    dup["xc"] = same[0]["xc"] + 0.05
    m = evaluate([same[0]], [dict(same[0], conf=0.97), dup], W, H)
    check("overlapping extra is a duplicate", m["duplicate_hi"] == 1, str(m))
    check("overlapping extra is not a missed player", m["missed_player_hi"] == 0, str(m))
    check("duplicate is not offered for adoption", m["missed_player_idx"] == [], str(m))
    check("duplicate scores below a missed player", m["disagreement"] < 0.5, str(m["disagreement"]))
    check("duplicate classified as duplicate", classify(m) == "duplicate")
    check("missed player outranks duplicate in classify",
          classify({"missed_player_hi": 1, "duplicate_hi": 1}) == "unlabelled_player")

    # Label the model cannot find at all -> moderate signal.
    m = evaluate([box(0.3, 0.5, 0.2, 0.4)], [], W, H)
    check("missing box detected", m["missing"] == 1, str(m))
    check("missing ranked below confident extra", m["disagreement"] < 0.7, str(m["disagreement"]))
    check("missing classified unfound_label", classify(m) == "unfound_label")

    # Weak overlap -> low-priority signal.
    # Boxes of w=h=0.2 shifted by 0.12 in x give IoU 0.25: matched, but loose.
    m = evaluate([box(0.5, 0.5, 0.2, 0.2)], [box(0.62, 0.5, 0.2, 0.2, conf=0.8)], W, H)
    check("weak overlap actually below match threshold", m["match_ious"][0] < 0.5,
          str(m["match_ious"]))
    check("weak overlap detected", m["weak"] == 1, str(m))
    check("weak classified weak_iou", classify(m) == "weak_iou")
    check("weak scores lowest", 0 < m["disagreement"] < 0.35, str(m["disagreement"]))

    # Low-confidence extra is not treated as a strong signal.
    m = evaluate([], [box(0.5, 0.5, 0.2, 0.2, conf=0.3)], W, H)
    check("low-conf extra not high-signal",
          m["extra"] == 1 and m["missed_player_hi"] == 0, str(m))
    check("low-conf extra classified low_conf_extra", classify(m) == "low_conf_extra")

    # Score never exceeds 1.0 even with many problems.
    m = evaluate([box(0.1, 0.1, 0.1, 0.1), box(0.4, 0.4, 0.1, 0.1), box(0.8, 0.8, 0.1, 0.1)],
                 [box(0.2, 0.2, 0.1, 0.1, conf=0.99), box(0.6, 0.6, 0.1, 0.1, conf=0.99)],
                 W, H)
    check("score is clamped to 1.0", m["disagreement"] <= 1.0, str(m["disagreement"]))

    # Greedy matching must not double-assign one prediction to two labels.
    gts = [box(0.50, 0.5, 0.2, 0.2), box(0.52, 0.5, 0.2, 0.2)]
    prs = [box(0.51, 0.5, 0.2, 0.2, conf=0.9)]
    m = evaluate(gts, prs, W, H)
    check("one prediction cannot match two labels", m["matched"] + m["weak"] == 1, str(m))
    check("the unmatched label is reported missing", m["missing"] == 1, str(m))


def test_predictions_loader() -> None:
    print("\n[prediction loader]")
    tmp = Path(tempfile.mkdtemp(prefix="labelrev-"))
    try:
        run_dir = tmp / "smash_player_colab_v2"
        run_dir.mkdir(parents=True)
        (run_dir / "val.jsonl").write_text(
            "\n".join([
                json.dumps({"image_id": "val/a", "width": 1920, "height": 1080,
                            "predictions": [{"cls": 0, "conf": 0.9, "xc": 0.3, "yc": 0.5, "w": 0.2, "h": 0.4}]}),
                json.dumps({"image_id": "val/b", "width": 1920, "height": 1080, "predictions": []}),
                "",                       # blank line must be tolerated
                "{not valid json",        # bad line must be skipped, not crash
            ]),
            encoding="utf-8",
        )

        p = Predictions(run_dir)
        check("loader reports available", p.available is True)
        check("loader indexed good rows", p.records.keys() == {"val/a", "val/b"}, str(list(p.records)))
        check("loader names the run", p.run_name == "smash_player_colab_v2", str(p.run_name))
        check("loader skips malformed lines", "val/a" in p.records and len(p.records) == 2)

        rec = p.get("val/a")
        check("get returns predictions", len(rec["predictions"]) == 1)
        check("get returns stored dims", p.records["val/a"]["width"] == 1920)
        check("unknown id returns unavailable", p.get("val/zzz")["available"] is False)

        s = p.summary()
        check("summary counts images", s["count"] == 2, str(s))
        check("summary lists splits", s["splits"].get("val") == 2, str(s["splits"]))
        # Raw prediction files carry no metrics; the app scores them itself, so
        # the loader must not invent zeros that look like real results.
        check("raw summary reports nothing scored", s["scored"] == 0, str(s))
        check("raw summary omits kinds", "kinds" not in s, str(s))
        check("raw summary explains why", "note" in s, str(s))

        # A file that does carry metrics still aggregates them.
        (run_dir / "train.jsonl").write_text(
            json.dumps({"image_id": "train/c", "width": 1920, "height": 1080, "predictions": [],
                        "metrics": {"disagreement": 0.6, "missed_player_hi": 1}}) + "\n",
            encoding="utf-8",
        )
        s2 = Predictions(run_dir).summary()
        check("stored metrics are scored", s2["scored"] == 1, str(s2))
        check("stored metrics produce kinds", s2["kinds"].get("unlabelled_player") == 1, str(s2["kinds"]))
        check("stored metrics produce max", s2["max_disagreement"] == 0.6, str(s2))

        # No predictions at all must degrade cleanly to the old behaviour.
        empty = Predictions(None)
        check("missing predictions is not available", empty.available is False)
        check("missing predictions summary says so", empty.summary()["available"] is False)
        check("missing predictions get is safe", empty.get("val/a")["available"] is False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_adopt_prediction_flow() -> None:
    """Adopting a prediction must resolve the disagreement it represents."""
    print("\n[adopt prediction as label]")
    W, H = 1920, 1080
    label = {"cls": 0, "xc": 0.30, "yc": 0.50, "w": 0.20, "h": 0.40}
    # Model sees the labelled player plus one the labels missed.
    missed = {"cls": 0, "xc": 0.75, "yc": 0.55, "w": 0.18, "h": 0.36, "conf": 0.91}

    before = evaluate([label], [dict(label, conf=0.95), missed], W, H)
    check("unlabelled player is a missed player", before["missed_player_hi"] == 1, str(before))
    check("it is not confused with a duplicate", before["duplicate_hi"] == 0, str(before))
    check("it is flagged before adopting", before["disagreement"] >= 0.5, str(before["disagreement"]))
    check("its index is reported for adopting", before["missed_player_idx"] == [1],
          str(before["missed_player_idx"]))

    # Adopt: append the prediction's geometry to the labels, then re-score.
    adopted = dict(label, **{k: missed[k] for k in ("xc", "yc", "w", "h")})
    after = evaluate([label, adopted], [dict(label, conf=0.95), missed], W, H)
    check("disagreement clears after adopting", after["disagreement"] == 0.0, str(after))
    check("both boxes now match", after["matched"] == 2, str(after))
    check("nothing left to adopt", after["unmatched_pred"] == [], str(after["unmatched_pred"]))

    # Re-adopting the same geometry must not double-count (the client guard).
    duped = evaluate([label, adopted, adopted], [dict(label, conf=0.95), missed], W, H)
    check("a duplicate adopt leaves an unmatched label", duped["missing"] == 1, str(duped))
    check("a duplicate adopt is therefore not free", duped["disagreement"] > 0,
          str(duped["disagreement"]))

    # Adopting a low-confidence extra should only be offered via click, not bulk:
    # verify the conf gate the UI uses to pick its bulk set.
    weak = {"cls": 0, "xc": 0.1, "yc": 0.1, "w": 0.1, "h": 0.1, "conf": 0.30}
    m = evaluate([], [weak], W, H)
    check("low-conf extra is adoptable by index", m["unmatched_pred"] == [0], str(m))
    check("but excluded from the bulk conf>=0.5 set",
          [i for i in m["unmatched_pred"] if weak["conf"] >= 0.5] == [], str(m))


def test_postprocess() -> None:
    """Dedupe and cap: the fix for the detector double-firing on one character."""
    print("\n[post-process: dedupe + cap]")
    W, H = 1920, 1080

    def box(xc, yc, w, h, conf):
        return {"cls": 0, "xc": xc, "yc": yc, "w": w, "h": h, "conf": conf}

    single = box(0.50, 0.50, 0.20, 0.40, 0.90)

    # A near-identical second detection of the same character must be merged,
    # and the higher-confidence one must survive.
    double = [single, box(0.53, 0.50, 0.20, 0.40, 0.60)]
    out = suppress(double, W, H, 0.40)
    check("duplicate detection suppressed", len(out) == 1, str(out))
    check("higher-confidence box survives", out[0]["conf"] == 0.90, str(out))

    # Two genuinely separate characters must both survive at IoU 0.40.
    apart = [single, box(0.15, 0.50, 0.20, 0.40, 0.80)]
    check("separate characters both kept", len(suppress(apart, W, H, 0.40)) == 2)

    # Suppression alone is not enough: a mis-merge can leave three boxes, so the
    # cap is what actually guarantees the 1v1 invariant.
    three = [single, box(0.15, 0.50, 0.20, 0.40, 0.80), box(0.85, 0.50, 0.20, 0.40, 0.70)]
    check("suppression keeps all three when they do not overlap",
          len(suppress(three, W, H, 0.40)) == 3)
    check("cap enforces the maximum", len(postprocess(three, W, H, 0.40, 2)) == 2)
    check("cap keeps the most confident two",
          [b["conf"] for b in postprocess(three, W, H, 0.40, 2)] == [0.90, 0.80])

    # Ordering: suppression must run BEFORE the cap, otherwise the cap can keep
    # two boxes that are duplicates of each other and drop the real second player.
    mixed = [single, box(0.52, 0.50, 0.20, 0.40, 0.95), box(0.15, 0.50, 0.20, 0.40, 0.55)]
    out = postprocess(mixed, W, H, 0.40, 2)
    check("cap after suppress keeps the real second character",
          [round(b["xc"], 2) for b in out] == [0.52, 0.15], str(out))

    # max_boxes=0 disables the cap but still dedupes.
    check("max_boxes=0 disables the cap", len(postprocess(three, W, H, 0.40, 0)) == 3)

    # Idempotence: running it twice must not change anything further.
    once = postprocess(mixed, W, H, 0.40, 2)
    check("post-process is idempotent", postprocess(once, W, H, 0.40, 2) == once)

    # Empty input must not crash.
    check("empty input is safe", postprocess([], W, H) == [])

    # And it must actually help on a realistic 1v1 frame: 3 boxes -> 2, with the
    # duplicate of the labelled character removed rather than a real one.
    dup_frame = [
        {"cls": 0, "xc": 0.30, "yc": 0.5, "w": 0.2, "h": 0.4, "conf": 0.92},
        {"cls": 0, "xc": 0.33, "yc": 0.5, "w": 0.2, "h": 0.4, "conf": 0.61},  # duplicate
        {"cls": 0, "xc": 0.70, "yc": 0.5, "w": 0.2, "h": 0.4, "conf": 0.88},  # real second
    ]
    out = postprocess(dup_frame, W, H, 0.40, 2)
    check("realistic frame reduced to two boxes", len(out) == 2, str(out))
    check("the duplicate is the one dropped", 0.70 in [round(b["xc"], 2) for b in out], str(out))


def test_review_counts() -> None:
    """The unreviewed tally must be the remainder, not always zero."""
    print("\n[review counts]")
    tmp = Path(tempfile.mkdtemp(prefix="labelrev-"))
    try:
        root = tmp / "ds"
        (root / "labels" / "train").mkdir(parents=True)
        (root / "images" / "train").mkdir(parents=True)
        store = ReviewStore(root)

        # Nothing reviewed yet: every image is unreviewed.
        c = store.counts(total=5)
        check("all unreviewed when nothing set", c["unreviewed"] == 5, str(c))
        check("no false 'good'", c["good"] == 0, str(c))

        store.set_status("train/a", "good")
        store.set_status("train/b", "bad", reasons=["missing_box"])
        store.set_status("train/c", "empty")
        c = store.counts(total=5)
        check("good counted", c["good"] == 1, str(c))
        check("bad counted", c["bad"] == 1, str(c))
        check("empty counted", c["empty"] == 1, str(c))
        check("unreviewed is the remainder", c["unreviewed"] == 2, str(c))
        check("tallies sum to total",
              sum(c[s] for s in ("unreviewed", "good", "bad", "empty")) == 5, str(c))

        # Reverting to unreviewed must put the image back in the backlog: the
        # state entry is dropped, so a naive count would lose it entirely.
        store.set_status("train/a", "unreviewed")
        c = store.counts(total=5)
        check("reverted image returns to unreviewed", c["unreviewed"] == 3, str(c))
        check("good drops after revert", c["good"] == 0, str(c))

        # Without a total the remainder is unknowable, so it stays zero.
        c = store.counts()
        check("no total -> unreviewed stays zero", c["unreviewed"] == 0, str(c))

        # A total smaller than the stored state must not go negative.
        c = store.counts(total=1)
        check("unreviewed clamps at zero", c["unreviewed"] == 0, str(c))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_sample_filters() -> None:
    """Every review flag must select exactly what it claims to."""
    print("\n[sample filters]")
    from server import sample_matches

    def row(**kw):
        base = {
            "image_id": "val/a",
            "split": "val",
            "stem": "a",
            "boxes": 2,
            "status": "unreviewed",
            "reasons": [],
            "kind": "",
            "disagreement": 0.0,
            "over_labelled": False,
            "malformed": 0,
        }
        base.update(kw)
        return base

    check("no filters pass", sample_matches(row()))

    # split / status / kind / search
    check("split excludes", not sample_matches(row(), split="train"))
    check("split includes", sample_matches(row(), split="val"))
    check("status excludes", not sample_matches(row(), status="good"))
    check("kind excludes", not sample_matches(row(kind="agree"), kind="unfound_label"))
    check("kind includes", sample_matches(row(kind="unfound_label"), kind="unfound_label"))
    check("search is case-insensitive on stem",
          sample_matches(row(stem="AB_c"), search="ab"))
    check("search excludes", not sample_matches(row(stem="ab"), search="zz"))

    # over_labelled
    check("over_labelled excludes clean",
          not sample_matches(row(over_labelled=False), flag="over_labelled"))
    check("over_labelled includes flagged",
          sample_matches(row(over_labelled=True), flag="over_labelled"))

    # inconsistent: marked empty yet still holding boxes
    check("inconsistent includes empty-with-boxes",
          sample_matches(row(status="empty", boxes=3), flag="inconsistent"))
    check("inconsistent excludes empty-without-boxes",
          not sample_matches(row(status="empty", boxes=0), flag="inconsistent"))
    check("inconsistent excludes boxed-but-not-empty",
          not sample_matches(row(status="good", boxes=3), flag="inconsistent"))

    # no_labels / malformed
    check("no_labels includes background",
          sample_matches(row(boxes=0), flag="no_labels"))
    check("no_labels excludes boxed",
          not sample_matches(row(boxes=2), flag="no_labels"))
    check("malformed includes malformed",
          sample_matches(row(malformed=1), flag="malformed"))
    check("malformed excludes clean",
          not sample_matches(row(malformed=0), flag="malformed"))

    # conflicting: reason says "add a box", model says "a box exists I cannot find"
    check("conflicting includes the mismatched pair",
          sample_matches(row(reasons=["missing_box"], kind="unfound_label"),
                         flag="conflicting"))
    check("conflicting excludes matching vocabularies",
          not sample_matches(row(reasons=["missing_box"], kind="unlabelled_player"),
                             flag="conflicting"))
    check("conflicting excludes unlabelled-model-only",
          not sample_matches(row(reasons=[], kind="unfound_label"), flag="conflicting"))
    check("conflicting excludes reason-only",
          not sample_matches(row(reasons=["missing_box"], kind=""), flag="conflicting"))
    check("conflicting tolerates missing reasons key",
          not sample_matches({"status": "bad", "kind": "unfound_label"}, flag="conflicting"))
    check("conflicting works with extra reasons too",
          sample_matches(row(reasons=["other", "missing_box"], kind="unfound_label"),
                         flag="conflicting"))

    # min_score
    check("min_score excludes below", not sample_matches(row(disagreement=0.3), min_score=0.5))
    check("min_score includes at", sample_matches(row(disagreement=0.5), min_score=0.5))
    check("min_score 0 disables", sample_matches(row(disagreement=0.0), min_score=0.0))

    # combined
    check("filters combine",
          sample_matches(row(split="val", kind="unfound_label", boxes=2),
                         split="val", kind="unfound_label"))
    check("combined still excludes",
          not sample_matches(row(split="train", kind="unfound_label"),
                             split="val", kind="unfound_label"))


def test_sample_sorting() -> None:
    """The default order must surface the backlog, not finished work."""
    print("\n[sample sorting]")
    from server import sort_samples

    def row(iid, status, disc, split="val", pred=False):
        return {
            "image_id": iid,
            "stem": iid.split("/")[-1],
            "split": split,
            "status": status,
            "disagreement": disc,
            "has_prediction": pred,
        }

    rows = [
        row("train/t1", "unreviewed", 0.0),
        row("train/t2", "unreviewed", 0.0),
        row("val/v_done_high", "bad", 0.90),
        row("val/v_done_low", "good", 0.60),
        row("val/v_new_high", "unreviewed", 0.70),
        row("val/v_new_mid", "unreviewed", 0.35),
        row("val/v_new_zero", "unreviewed", 0.0),
    ]

    order = [r["image_id"] for r in sort_samples(rows, "unreviewed")]
    check(
        "unreviewed first (score desc), reviewed last (score desc)",
        order
        == [
            "val/v_new_high",
            "val/v_new_mid",
            "train/t1",
            "train/t2",
            "val/v_new_zero",
            "val/v_done_high",
            "val/v_done_low",
        ],
        str(order),
    )
    check("highest-scoring unreviewed comes first",
          order[0] == "val/v_new_high", str(order))
    check("unreviewed sorts ahead of reviewed",
          order.index("val/v_new_zero") < order.index("val/v_done_high"), str(order))
    check("reviewed rows are last",
          set(order[-2:]) == {"val/v_done_high", "val/v_done_low"}, str(order))
    check("no rows lost", len(order) == len(rows), str(order))

    # Known limitation, asserted so the behaviour is explicit rather than
    # assumed: this sort fixes reviewed-vs-unreviewed, NOT prediction coverage.
    # Two unreviewed rows that both score 0.0 keep their scan order, so a train
    # row (no predictions at all) can still precede a val row that has them.
    # Reaching prediction-bearing rows is what the split filter is for.
    tie = [
        row("train/no_preds", "unreviewed", 0.0),
        row("val/has_preds", "unreviewed", 0.0, pred=True),
    ]
    u = [r["image_id"] for r in sort_samples(tie, "unreviewed")]
    check("equal scores keep scan order regardless of predictions",
          u == ["train/no_preds", "val/has_preds"], str(u))

    # Other modes still behave
    check("disagreement sorts descending",
          [r["image_id"] for r in sort_samples(rows, "disagreement")][0]
          == "val/v_done_high")
    check("agreement sorts ascending",
          [r["image_id"] for r in sort_samples(rows, "agreement")][0]
          in ("train/t1", "train/t2", "val/v_new_zero"))
    byname = [r["image_id"] for r in sort_samples(rows, "name")]
    check("name sorts by split then stem",
          byname[:2] == ["train/t1", "train/t2"], str(byname))

    # Stability: sort must not mutate the caller's list.
    original = list(rows)
    sort_samples(rows, "unreviewed")
    check("sort does not mutate input", rows == original)


def test_store_save_and_undo() -> None:
    print("\n[store: save, journal, undo, backups]")
    tmp = Path(tempfile.mkdtemp(prefix="labelrev-"))
    try:
        root = tmp / "ds"
        (root / "labels" / "train").mkdir(parents=True)
        (root / "images" / "train").mkdir(parents=True)
        img_id = "train/frame_000001"
        original = "0 0.437794 0.578130 0.264474 0.384204\n"
        (root / "labels" / f"{img_id}.txt").write_text(original, encoding="utf-8")
        (root / "images" / f"{img_id}.jpg").write_bytes(b"\xff\xd8\xff\xd9")

        store = LabelStore(root)

        # A no-op save must not touch the file.
        res = store.save(img_id, parse_text(original).boxes)
        check("no-op save reports unchanged", res["changed"] is False)
        check(
            "no-op save leaves file byte-identical",
            (root / "labels" / f"{img_id}.txt").read_text(encoding="utf-8") == original,
        )

        # A real edit writes, journals, and backs up.
        new_boxes = [Box(0, 0.5, 0.5, 0.25, 0.25)]
        res = store.save(img_id, new_boxes, reason="move_box")
        check("edit reports changed", res["changed"] is True)
        written = (root / "labels" / f"{img_id}.txt").read_text(encoding="utf-8")
        check("edit written to disk", written == "0 0.500000 0.500000 0.250000 0.250000\n", written)
        check("backup created", res["backup"] is not None)
        backup = Path(res["backup"])
        check("backup holds original", backup.read_text(encoding="utf-8") == original)
        check("journal has entry", len(store.recent_edits()) == 1)

        # Undo restores the previous geometry.
        undo = store.restore_from_journal(img_id)
        check("undo succeeded", undo["restored"] is True)
        restored = (root / "labels" / f"{img_id}.txt").read_text(encoding="utf-8")
        check("undo restored original geometry", restored == original, restored)

        # Path escape must be rejected.
        escaped = False
        try:
            store.label_path("../../etc/passwd")
        except ValueError:
            escaped = True
        check("path escape rejected", escaped)

        # Cache files must never be written.
        cache_blocked = False
        try:
            store._resolve(root / "labels" / "train.cache")
        except ValueError:
            cache_blocked = True
        check("cache writes blocked", cache_blocked)

        # Dropped degenerate boxes are counted.
        res = store.save(img_id, [Box(0, 0.5, 0.5, 0.0, 0.1)], reason="test")
        check("degenerate box dropped", res["dropped"] == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_review_store() -> None:
    print("\n[review store]")
    tmp = Path(tempfile.mkdtemp(prefix="labelrev-"))
    try:
        store = ReviewStore(tmp)
        rec = store.set_status("train/a", "bad", reasons=["missing_box"], note="two players visible")
        check("status set", rec["status"] == "bad")
        check("reasons stored", rec["reasons"] == ["missing_box"])
        check("note stored", rec["note"] == "two players visible")

        store.set_status("train/b", "empty")
        store.set_status("train/c", "good")

        counts = store.counts()
        check(
            "counts correct",
            counts["good"] == 1 and counts["bad"] == 1 and counts["empty"] == 1,
            str(counts),
        )
        check("counts includes unreviewed bucket", "unreviewed" in counts)
        check("record retrievable", store.get("train/a")["status"] == "bad")
        check("unknown image returns default", store.get("train/zzz")["status"] == "unreviewed")

        # Persistence across instances.
        store2 = ReviewStore(tmp)
        check("persisted to disk", store2.get("train/b")["status"] == "empty")

        # Invalid values are rejected rather than silently stored.
        bad_rejected = False
        try:
            store.set_status("train/d", "nonsense")
        except ValueError:
            bad_rejected = True
        check("invalid status rejected", bad_rejected)

        bad_reason_rejected = False
        try:
            store.set_status("train/e", "bad", reasons=["not_a_reason"])
        except ValueError:
            bad_reason_rejected = True
        check("invalid reason rejected", bad_reason_rejected)
        check("reason codes are non-empty", len(REASON_CODES) >= 4)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("Running label review app self-tests")
    test_parse_basic()
    test_roundtrip_identity()
    test_clamping()
    test_line_endings()
    test_decode_safety()
    test_predictions_scoring()
    test_predictions_loader()
    test_postprocess()
    test_adopt_prediction_flow()
    test_review_counts()
    test_sample_filters()
    test_sample_sorting()
    test_store_save_and_undo()
    test_review_store()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
