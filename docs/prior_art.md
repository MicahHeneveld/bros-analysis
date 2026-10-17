# Prior art, licensing, and what we deliberately did not copy

Before building the label review app we surveyed existing tooling. The conclusion was
that **no existing project occupies our niche**: a zero-dependency local server that
edits YOLO `.txt` labels in place with review tracking. The space splits into
heavyweight apps (Qt/React/Electron), single-file HTML toys with no review state, or full
platforms with accounts and databases.

This document records what we looked at, **what its license permits**, and what we
actually reused. It exists so the decisions are auditable later.

## Why we built instead of adopting

**FiftyOne** ([voxel51/fiftyone](https://github.com/voxel51/fiftyone), Apache-2.0) is
genuinely good and worth revisiting, but it was rejected for this pass because it
requires Python 3.10 (we are on 3.9.6), pulls a ~2 GB install plus a bundled MongoDB
process, and would force a database → YOLO export round trip for what is fundamentally a
text-file editing job. Those are real costs for a dataset of 3,857 small files.

## License map

This decides what we may **copy**, not just read.

| Project | Stars | License | Use |
|---|---|---|---|
| [sicaizhuang/yolo-annotator-desktop](https://github.com/sicaizhuang/yolo-annotator-desktop) | 0 | **MIT** | Copied patterns (closest match to our safety/review design) |
| [franchesoni/minimalist-annotation](https://github.com/franchesoni/minimalist-annotation) | 4 | **MIT** | Copied canvas zoom/pan interaction ideas |
| [sPappalard/DatasetEngine](https://github.com/sPappalard/DatasetEngine) | 31 | **MIT** | Copied validation-check ideas |
| [HumanSignal/labelImg](https://github.com/HumanSignal/labelImg) | 25k | **MIT** | Reference; desktop Qt, unmaintained since 2024 |
| [developer0hye/Yolo_Label](https://github.com/developer0hye/Yolo_Label) | 709 | **MIT** | Reference for canvas UX |
| [flowhl/YoloAnnotationEditor](https://github.com/flowhl/YoloAnnotationEditor) | 6 | Apache-2.0 | Reference only (C#/WPF, Windows-only) |
| [voxel51/fiftyone](https://github.com/voxel51/fiftyone) | 11k | Apache-2.0 | Reference; revisit later for mistakenness scoring |
| [ManuelPopa/alice2](https://github.com/ManuelPopa/alice2) | 0 | CC BY-NC 4.0 | **Inspiration only — NonCommercial** |
| [CVHub520/X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) | 10.5k | GPL-3.0 | **Do not copy — copyleft** |
| [wkentaro/labelme](https://github.com/wkentaro/labelme) | 16k | GPL-3.0 | **Do not copy — copyleft** |
| [visionify/dataset-review](https://github.com/visionify/dataset-review) | 1 | No license | **Do not copy — all rights reserved** |

### Two findings that mattered

**`alice2` / `alice` is the closest architectural match** — a single-file Python +
Pillow web canvas editor — but it is **CC BY-NC 4.0 NonCommercial**. The project vision
includes a coaching platform and tournament integration, so copying that code would be a
licensing risk. Treat as inspiration only.

**`visionify/dataset-review` independently arrived at nearly the same API shape we
designed** (`GET /api/images?split=&page`, `GET/PUT /api/annotations/:split/:base`,
`PATCH /api/reviewed`, `GET /api/validation`). That was strong validation of the design,
but the repo has **no license**, so nothing was borrowed from it.

## Patterns adopted (from MIT-licensed sources)

`sicaizhuang/yolo-annotator-desktop` validated and sharpened the design:

- **Atomic autosave** on every change, plus undo/redo.
- **Recovery copy on first edit** of each label file per session.
- **Preserve malformed label rows** instead of silently dropping them. This was
  directly actionable: our own `clean_label_lines` in
  [scripts/prepare_cvat_import.py](../scripts/prepare_cvat_import.py) discards any row
  with fewer than 5 parts, which would destroy data if reused for editing.
- **Reviewed / labeled / empty filters**, with **explicitly reviewed-empty** as a
  first-class state.
- **Blocking errors vs non-blocking warnings** separated in validation output.
- **Export never edits sources**, and refuses destinations inside source folders.
- Keyboard conventions: `u` for next unreviewed, `n` for mark-reviewed-empty.
- Quality checks: corrupt images, invalid geometry, duplicate boxes, orphan labels.

`franchesoni/minimalist-annotation` informed the canvas zoom/pan interaction (scroll to
zoom at the cursor, drag to pan).

## Original work in this repo

The implementation here is newly written. Nothing was copied verbatim. The specific
things we designed ourselves, because the prior art had no answer for them:

- **Byte-identical no-op round trip**, which required reading label files as raw bytes
  rather than through `read_text` universal-newline translation.
- **CRLF preservation.** The real dataset uses Windows line endings. Naive rewriting
  would have converted every edited file to LF and produced whole-file diffs on Drive.
- **Non-UTF-8 files made read-only** rather than being edited into corruption.
- **The `--allow-cloud` guard**, which refuses to edit inside a synced Drive mount and
  points at `stage_dataset.py` instead.
- **Colab-aware sync guard**, which refuses `sync_labels.py` while a run looks active.

## Attribution

No code was copied, so no source-file attribution is required. This document records the
influence of the MIT-licensed projects above and the deliberate exclusion of the
non-permissive ones. If code is ever copied from any of these in future, the MIT and
Apache-2.0 notices must be carried into this repo at that time, and the GPL-3.0 and
CC BY-NC 4.0 projects must remain off-limits for this project.
