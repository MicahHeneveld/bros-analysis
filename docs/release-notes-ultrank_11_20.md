Frames for the `ultrank_11_20` dataset: gameplay-only samples from the UltRank
2026.1 #11-20 video, sampled at ~2 fps with menu/transition frames dropped.

## Fetching

```bash
python scripts/bounty.py fetch --dataset data/datasets/ultrank_11_20
```

That downloads this archive, verifies its size, and extracts the images into
`data/datasets/ultrank_11_20/images/`. Pass `--force` to overwrite an existing
image directory.

## What is not in here

Labels are **not** in this archive. They live in git under
`data/datasets/ultrank_11_20/labels/`, because they are the thing contributors
review and commit. Only images ship as a release asset.

## Contents

1654 PNG frames (1920x1080). Model predictions for the same frames are in
`data/datasets/ultrank_11_20/predictions/`.
