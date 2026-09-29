#!/bin/bash
# 误信息实验全量：4 组 × 243 题 × 3 采样，每组 2 切片共 8 进程并行
set -u
DIR="$HOME/misinfo实验_2026-09-13"
CODE="$DIR/code"; RES="$DIR/results"; LOGS="$DIR/logs"
export EVAL_API_KEY="sk-YOUR_API_KEY"
export EVAL_MODEL="deepseek-v4-flash"
export EVAL_SAMPLES=3

GROUPLIST="base_original base_misinformed inst_misinformed protocol_misinformed"
for g in $GROUPLIST; do
  for p in 0 1; do
    s=$((p*122)); e=$((s+122)); [ "$e" -gt 243 ] && e=243
    python3 -u "$CODE/eval_misinfo.py" --group "$g" --start "$s" --end "$e" \
      --tag "run1_p${p}" --outdir "$RES" > "$LOGS/misinfo_${g}_p${p}.log" 2>&1 &
  done
done
wait
echo "MISINFO ALL DONE"
