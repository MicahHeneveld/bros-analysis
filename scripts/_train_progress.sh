#!/usr/bin/env bash
# Compact progress reporter for a local training run.
# Usage: bash scripts/_train_progress.sh [run_name] [epochs]
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
RUN_NAME="${1:-smash_player_colab_v4}"
TOTAL="${2:-50}"
RUN=".staging/runs/$RUN_NAME"
LOG=".staging/${RUN_NAME##*_}_train.log"

echo "=== $RUN_NAME ==="
if pgrep -f "train_ssbu.py.*$RUN_NAME" >/dev/null 2>&1; then
  echo "status: RUNNING"
else
  echo "status: NOT RUNNING"
fi

if [ -f "$RUN/results.csv" ]; then
  n=$(( $(wc -l < "$RUN/results.csv") - 1 ))
  [ "$n" -lt 0 ] && n=0
  echo "epochs: $n / $TOTAL"
  echo "epoch,mAP50,mAP50-95,precision,recall"
  awk -F, 'NR>1 && $1!="" {printf "%s,%.4f,%.4f,%.4f,%.4f\n", $1,$8,$9,$6,$7}' "$RUN/results.csv" | tail -5
  awk -F, 'NR>1 && $9>m {m=$9; e=$1} END {if(e!="") printf "best mAP50-95: %.4f (epoch %d)\n", m, e}' "$RUN/results.csv"
else
  echo "results.csv: not created yet"
fi

echo
echo "latest in-epoch line:"
tr '\r' '\n' < "$LOG" 2>/dev/null | grep -E "/$TOTAL " | tail -1 | tr -s ' ' | cut -c1-140
