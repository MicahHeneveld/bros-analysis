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


def test_ingest_gating() -> None:
    """The gameplay gate is what decides which frames reach the dataset.

    It is pure logic over detection boxes, so it is worth pinning down: a
    regression here silently either floods the review queue with menus or
    throws away real gameplay.
    """
    print("\n[ingest gating]")
    import argparse
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ingest_video",
        Path(__file__).resolve().parent.parent / "scripts" / "ingest_video.py",
    )
    ingest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingest)

    def args(**kw):
        base = dict(player_conf=0.35, min_height=0.04, max_height=0.90, min_aspect=0.45)
        base.update(kw)
        return argparse.Namespace(**base)

    # A believable fighter: tall, confident, a sensible slice of the frame.
    fighter = {"conf": 0.9, "xc": 0.3, "yc": 0.5, "w": 0.08, "h": 0.22}
    # The detector firing twice on one character (same shape, high confidence).
    second = {"conf": 0.8, "xc": 0.7, "yc": 0.5, "w": 0.08, "h": 0.22}

    check("fighter passes the gate", ingest.is_plausible_player(fighter, 1080, 1920, args()))
    check(
        "two fighters count as two players",
        ingest.count_players([fighter, second], 1080, 1920, args()) == 2,
    )

    low = dict(fighter, conf=0.20)
    check("low-confidence box does not count", not ingest.is_plausible_player(low, 1080, 1920, args()))

    tiny = dict(fighter, h=0.01)
    check("sliver box rejected (UI element)", not ingest.is_plausible_player(tiny, 1080, 1920, args()))

    huge = dict(fighter, h=0.95)
    check("near-full-frame box rejected", not ingest.is_plausible_player(huge, 1080, 1920, args()))

    squat = dict(fighter, w=0.40, h=0.10)
    check("wide flat box rejected (a banner, not a fighter)", not ingest.is_plausible_player(squat, 1080, 1920, args()))

    check(
        "menu frame with no detections is gated out",
        ingest.count_players([], 1080, 1920, args()) == 0,
    )

    # The no-op / empty distinction drives the process exit code.
    check("run that saved frames exits 0", ingest.exit_code_for({"saved": 5, "already_present": 0}) == 0)
    check(
        "fully-present re-run exits 0 (not an error)",
        ingest.exit_code_for({"saved": 0, "already_present": 9}) == 0,
    )
    check(
        "run that kept nothing and had nothing exits 1",
        ingest.exit_code_for({"saved": 0, "already_present": 0}) == 1,
    )

    check(
        "dedupe signature of identical frames is zero",
        float(ingest.thumbnail(__import__("numpy").zeros((100, 100, 3), dtype="uint8")).sum()) == 0.0,
    )


def test_distance_metric() -> None:
    """Pin the geometry against a synthetic camera with known ground truth.

    The whole point of the body-lengths metric is that it cancels zoom, and of
    the ground-plane model that it recovers depth. Both are claims that are easy
    to get subtly wrong, so they are checked against a camera whose parameters
    are known exactly rather than against recorded output.
    """
    print("\n[distance]")
    import math

    from distance import DEFAULT_G, body_lengths, calibrate, measure

    W, H = 1920, 1080
    G = DEFAULT_G
    F = G * H          # focal length in pixels
    CAM_H = 2.5        # camera height above the floor, in fighter-heights
    HORIZON = 0.32 * H

    def to_box(X: float, Z: float) -> dict:
        """Project a fighter standing on the floor at world (X, Z)."""
        x = W / 2 + F * X / Z
        y = HORIZON + F * CAM_H / Z
        hpx = F / Z
        return {"xc": x / W, "yc": (y - hpx / 2) / H, "w": (hpx * 0.7) / W, "h": hpx / H, "cls": 0}

    # Same depth: body-lengths must be exact, and both metrics must agree.
    a, b = to_box(0.0, 12.0), to_box(4.0, 12.0)
    body, ratio = body_lengths(a, b, W, H)
    check("same-depth body-lengths is exact", abs(body - 4.0) < 1e-6, f"got {body:.4f}")
    check("same-depth ratio is 1", abs(ratio - 1.0) < 1e-6, f"got {ratio:.4f}")

    # A zoom change must not move the body-lengths reading. Same separation at
    # half the depth means the camera zoomed in 2x; the answer is unchanged.
    az, bz = to_box(0.0, 6.0), to_box(4.0, 6.0)
    body_z, _ = body_lengths(az, bz, W, H)
    check("body-lengths is invariant to zoom", abs(body_z - body) < 1e-6,
          f"{body:.4f} at Z=12 vs {body_z:.4f} at Z=6")

    # Calibration must recover the camera it was generated from.
    pairs = [
        (to_box(x1, z1), to_box(x2, z2))
        for x1, z1, x2, z2 in [
            (-3.0, 10.0, 2.0, 11.0), (1.0, 12.0, 5.0, 12.5),
            (-1.0, 9.0, 3.5, 9.5), (0.5, 13.0, 4.0, 13.2),
            (-2.0, 11.0, 1.5, 11.4), (2.0, 10.5, 6.0, 10.8),
            (-4.0, 12.0, 0.0, 12.1), (1.5, 11.5, 5.5, 11.9),
        ] * 6  # enough points to clear MIN_CALIBRATION_SAMPLES
    ]
    cal = calibrate(pairs, W, H)
    check("calibration succeeds on exact data", cal is not None)
    if cal:
        check("calibration recovers the horizon", abs(cal.horizon_norm - HORIZON / H) < 0.01,
              f"{cal.horizon_norm:.4f} vs {HORIZON / H:.4f}")
        check("calibration recovers camera height", abs(cal.camera_height - CAM_H) < 0.05,
              f"{cal.camera_height:.3f} vs {CAM_H:.3f}")
        check("calibration fit is strong", cal.r2 > 0.99, f"R2={cal.r2:.3f}")

        # Ground plane must recover pure depth separation, which body-lengths
        # cannot see at all. This is the whole reason the model exists.
        d1, d2 = to_box(0.0, 10.0), to_box(0.0, 15.0)
        m = measure(d1, d2, W, H, cal)
        check("ground recovers pure depth separation",
              m.ground_lengths is not None and abs(m.ground_lengths - 5.0) < 0.05,
              f"got {m.ground_lengths}")
        body_depth, _ = body_lengths(d1, d2, W, H)
        check("body-lengths badly under-reports pure depth", body_depth < 2.0,
              f"{body_depth:.2f} against a true 5.0")
        check("depth component dominates for depth-only pair",
              (m.depth or 0) > (m.lateral or 0))

        # An unreliable fit must not be used: better to omit than to guess.
        bad = calibrate(pairs, W, H)
        if bad:
            bad.reliable = False
            m2 = measure(d1, d2, W, H, bad)
            check("unreliable calibration yields no ground estimate",
                  m2.ground_lengths is None)

    # Without a calibration only the scale-free metric is available.
    m3 = measure(a, b, W, H, None)
    check("no calibration => no ground estimate", m3.ground_lengths is None)
    check("body-lengths always available", abs(m3.body_lengths - 4.0) < 1e-6)

    # Degenerate boxes must not raise or divide by zero.
    flat = {"xc": 0.5, "yc": 0.5, "w": 0.1, "h": 0.0, "cls": 0}
    check("zero-height box does not raise",
          math.isfinite(body_lengths(flat, a, W, H)[0]))


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


def test_video_mode() -> None:
    """Video playback depends on three pieces of logic that are easy to break.

    Image ids are built from video filenames, which contain '#', spaces and
    parentheses. That broke every image in the first ingested dataset, so the
    slug and the percent-decoding are pinned here. Range parsing is what lets
    the browser seek instead of downloading a 600 MB file before playing.
    """
    print("\n[video mode: slugs, ranges, timeline]")
    import importlib.util

    import server

    spec = importlib.util.spec_from_file_location(
        "ingest_video",
        Path(__file__).resolve().parent.parent / "scripts" / "ingest_video.py",
    )
    ingest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingest)

    # A realistic hostile title: spaces, '#', parentheses, '&'.
    slug = ingest.slugify("Top 50 Smash Ultimate Players - UltRank 2026.1 #11-20 - Name (1080p)")
    check("slug has no '#'", "#" not in slug, slug)
    check("slug has no spaces", " " not in slug, slug)
    check("slug is URL-safe", all(c.isalnum() or c in "-_" for c in slug), slug)
    check("slug keeps the useful words", "UltRank_2026_1_11-20" in slug, slug)
    check("slug is bounded", len(slug) <= 60, str(len(slug)))
    check(
        "slug never collapses to nothing",
        ingest.slugify("###") == "video",
        ingest.slugify("###"),
    )

    # Percent-encoded ids must decode back to the on-disk stem.
    from urllib.parse import quote, unquote

    hostile = "ultrank/Top 50 #11-20 (1080p)_frame_000001"
    encoded = quote(hostile, safe="/")
    check("encoded id has no raw '#'", "#" not in encoded, encoded)
    check("encoded id round-trips", unquote(encoded) == hostile)

    # Range parsing: the video element issues exactly these shapes.
    size = 1000
    check("no header means whole file", server.parse_range(None, size) == (0, 999, False))
    check("open-ended range", server.parse_range("bytes=900-", size) == (900, 999, True))
    check("bounded range", server.parse_range("bytes=0-99", size) == (0, 99, True))
    check("suffix range", server.parse_range("bytes=-100", size) == (900, 999, True))
    check("end past EOF is clamped", server.parse_range("bytes=990-5000", size) == (990, 999, True))
    check("start at EOF is unsatisfiable", server.parse_range("bytes=1000-", size) is None)
    check("inverted range is unsatisfiable", server.parse_range("bytes=500-100", size) is None)
    check("garbage falls back to whole file", server.parse_range("bytes=abc", size) == (0, 999, False))

    tmp = Path(tempfile.mkdtemp(prefix="labelrev-video-"))
    try:
        root = tmp / "ds"
        (root / "labels" / "clip").mkdir(parents=True)
        (root / "images" / "clip").mkdir(parents=True)
        preds_dir = root / "predictions"
        preds_dir.mkdir()
        video = tmp / "clip.mp4"
        video.write_bytes(b"not really an mp4")

        # Three kept frames out of a sparse timeline, deliberately out of order
        # in the file so the timeline has to sort by time.
        times = {"frame_000003": 6.0, "frame_000001": 2.0, "frame_000002": 4.0}
        lines = []
        for stem, t in times.items():
            image_id = f"clip/{stem}"
            (root / "images" / f"{image_id}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (root / "labels" / f"{image_id}.txt").write_text(
                "0 0.250000 0.500000 0.100000 0.200000\n", encoding="utf-8"
            )
            lines.append(
                {
                    "image_id": image_id,
                    "width": 1920,
                    "height": 1080,
                    "time_s": t,
                    "frame_index": int(stem.split("_")[-1]),
                    "source": "clip",
                    "predictions": [
                        {"cls": 0, "conf": 0.9, "xc": 0.3, "yc": 0.5, "w": 0.1, "h": 0.2}
                    ],
                }
            )
        (preds_dir / "clip.jsonl").write_text(
            "\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8"
        )
        (root / "manifest.json").write_text(
            json.dumps({"sources": {"clip": {"path": str(video), "duration_s": 10.0}}}),
            encoding="utf-8",
        )

        state = server.AppState(root, preds_dir)
        check("manifest video is discovered", len(state.videos) == 1, str(len(state.videos)))
        if state.videos:
            check("discovered video name", state.videos[0]["name"] == "clip", state.videos[0]["name"])

        tl = state.video_timeline(0)
        frames = tl["frames"]
        check("timeline covers every sample", len(frames) == 3, str(len(frames)))
        check(
            "timeline is sorted by time",
            [f["t"] for f in frames] == [2.0, 4.0, 6.0],
            str([f["t"] for f in frames]),
        )
        check("timeline carries label boxes", len(frames[0]["labels"]) == 1)
        check("timeline carries model boxes", len(frames[0]["model"]) == 1)
        check("labels expose cls", frames[0]["labels"][0]["cls"] == 0)

        bad = False
        try:
            state.video_timeline(99)
        except ValueError:
            bad = True
        check("out-of-range video index rejected", bad)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bounty_metadata() -> None:
    """Bounty points are computed once, at creation, and must round-trip."""
    print("\n[bounty metadata]")
    scripts = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import bounty as bounty_mod
    finally:
        sys.path.remove(str(scripts))

    meta = {
        "dataset": "ssbu_players",
        "split": "train",
        "target": 250,
        "filter": {"kind": "unlabelled_player"},
        "difficulty": 1.31,
        "points": 328,
    }
    body = f"Some prose.\n\n{bounty_mod.dump_meta(meta)}\n"
    parsed = bounty_mod.parse_meta(body)
    check("metadata round-trips", parsed == meta, str(parsed))
    check("prose is not metadata", bounty_mod.parse_meta("just text") is None)
    check("broken JSON is ignored",
          bounty_mod.parse_meta("<!-- bounty: {nope} -->") is None)

    # The HTML comment must stay invisible when GitHub renders the issue.
    check("metadata is an HTML comment",
          bounty_mod.dump_meta(meta).startswith("<!--")
          and bounty_mod.dump_meta(meta).endswith("-->"))

    # Points round outward: a bounty is never worth less than advertised.
    check("points round up", bounty_mod.points_for(250, 1.31) == 328,
          str(bounty_mod.points_for(250, 1.31)))
    check("points round up on .001", bounty_mod.points_for(100, 1.001) == 101,
          str(bounty_mod.points_for(100, 1.001)))

    # Only an actual claim counts. A long comment that merely uses the word
    # "claim" must not make a bounty look taken — a 22KB spam comment did.
    prose = "Some long spam body that says you can claim a reward here."
    check("prose mentioning claim is not a claim",
          not bounty_mod.CLAIM_RE.findall(prose))
    check("real claim is detected",
          bounty_mod.CLAIM_RE.findall("claim by @someone") == ["someone"])
    check("claim marker is case tolerant",
          bounty_mod.CLAIM_RE.findall("Claim by @Someone") == ["Someone"])
    check("claim inside a paragraph is ignored",
          not bounty_mod.CLAIM_RE.findall("thanks!\nclaim by @x was wrong\n"))

    tmp = Path(tempfile.mkdtemp(prefix="bounty-"))
    try:
        # A dataset with no non-empty labels is a bootstrap job, not a review
        # job; the bounty body has to say so or it misdescribes the work.
        ds = tmp / "ds"
        (ds / "labels" / "train").mkdir(parents=True)
        (ds / "images" / "train").mkdir(parents=True)
        for name in ("a", "b"):
            (ds / "labels" / "train" / f"{name}.txt").write_text("")
        check("empty dataset counts as unlabelled",
              bounty_mod.labelled_frames(ds) == 0, str(bounty_mod.labelled_frames(ds)))

        (ds / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        check("one labelled frame detected",
              bounty_mod.labelled_frames(ds) == 1, str(bounty_mod.labelled_frames(ds)))

        check("missing labels dir is unlabelled",
              bounty_mod.labelled_frames(tmp / "nope") == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    tmp = Path(tempfile.mkdtemp(prefix="bounty-"))
    try:
        # Confident, single-detection frames are easy; empty and weak frames are
        # not, and the multiplier has to reflect that.
        easy = tmp / "easy.jsonl"
        easy.write_text("\n".join(
            json.dumps({"predictions_clean": [{"conf": 0.97}] * 2}) for _ in range(4)
        ) + "\n")
        hard = tmp / "hard.jsonl"
        hard.write_text("\n".join(
            json.dumps({"predictions_clean": []}) for _ in range(4)
        ) + "\n")

        d_easy = bounty_mod.slice_difficulty(easy, 4)
        d_hard = bounty_mod.slice_difficulty(hard, 4)
        check("confident slice is cheap", d_easy < 1.2, str(d_easy))
        check("empty slice costs more", d_hard > d_easy, f"{d_hard} vs {d_easy}")
        check("difficulty stays bounded", 1.0 <= d_hard <= 2.0, str(d_hard))
        check("missing predictions -> neutral", bounty_mod.slice_difficulty(None, 10) == 1.0)
        check("missing file -> neutral",
              bounty_mod.slice_difficulty(tmp / "nope.jsonl", 10) == 1.0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_image_bundle_extraction() -> None:
    """Bundles are zipped inconsistently; extraction must flatten either shape.

    Both bug shapes this guards against were real: unpack_archive returns None
    (so a truthiness check silently aborted every fetch), and a wrapping folder
    was reproduced under images/ instead of being stripped.
    """
    print("\n[image bundle extraction]")
    scripts = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import bounty as bounty_mod
    finally:
        sys.path.remove(str(scripts))

    tmp = Path(tempfile.mkdtemp(prefix="bundle-"))
    try:
        def make(root: Path, layout: str) -> Path:
            base = root / layout
            base.mkdir(parents=True)
            for name in ("a.jpg", "b.png"):
                (base / name).write_bytes(b"not-a-real-image")
            return base

        # Shape 1: archive root holds images/ directly.
        flat_root = tmp / "flat"
        make(flat_root, "images")
        found = bounty_mod._images_root(flat_root)
        check("bare images/ found", found == flat_root / "images", str(found))

        # Shape 2: dataset/images/ one level down.
        nested_root = tmp / "nested"
        make(nested_root, "ds/images")
        found = bounty_mod._images_root(nested_root)
        check("nested ds/images found", found == nested_root / "ds" / "images", str(found))

        # Shape 3: extra wrapper (bundle/ds/images), as produced by `zip -r`.
        wrapped_root = tmp / "wrapped"
        make(wrapped_root, "bundle/ds/images")
        found = bounty_mod._images_root(wrapped_root)
        check("wrapped bundle/.../images found",
              found == wrapped_root / "bundle" / "ds" / "images", str(found))

        # Shape 4: splits under images/. The `images` dir itself holds no files,
        # but it is still the dataset root — flattening it would destroy splits.
        split_root = tmp / "split"
        (split_root / "images" / "train").mkdir(parents=True)
        (split_root / "images" / "train" / "a.jpg").write_bytes(b"x")
        found = bounty_mod._images_root(split_root)
        check("images/ with split subdir is the root", found == split_root / "images", str(found))

        split_dest = tmp / "splitout" / "images"
        bounty_mod._copy_tree(found, split_dest)
        check("split directory preserved",
              (split_dest / "train" / "a.jpg").exists(),
              str(sorted(p.name for p in split_dest.rglob("*"))))

        # Non-image files must not be mistaken for the image root.
        junk = tmp / "junk"
        (junk / "docs").mkdir(parents=True)
        (junk / "docs" / "readme.txt").write_text("hi")
        check("no images -> None", bounty_mod._images_root(junk) is None)

        # Copying flattens into dest, so labels/ and images/ stay siblings.
        dest = tmp / "out" / "images"
        n = bounty_mod._copy_tree(wrapped_root / "bundle" / "ds" / "images", dest)
        check("copied both files", n == 2, str(n))
        check("landed flat in dest",
              (dest / "a.jpg").exists() and (dest / "b.png").exists(),
              str(sorted(p.name for p in dest.iterdir())))
        check("no nested wrapper under dest",
              not (dest / "bundle").exists(), str(sorted(p.name for p in dest.iterdir())))
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
    test_ingest_gating()
    test_distance_metric()
    test_review_counts()
    test_sample_filters()
    test_sample_sorting()
    test_store_save_and_undo()
    test_review_store()
    test_video_mode()
    test_bounty_metadata()
    test_image_bundle_extraction()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
