#!/bin/bash
# 4B 本地零样本守护循环：进程死亡自动断点重启，直到两任务各 200 行
DIR="$HOME/TCMEval-SDT显著性检验_2026-09-06"
while true; do
  s=$(wc -l < "$DIR/results/评估_zero_qwen4b_zero_run1_syndrome.jsonl" 2>/dev/null || echo 0)
  p=$(wc -l < "$DIR/results/评估_zero_qwen4b_zero_run1_pathogenesis.jsonl" 2>/dev/null || echo 0)
  if [ "$s" -ge 200 ] && [ "$p" -ge 200 ]; then echo "4B ALL DONE"; break; fi
  echo "[$(date +%m-%d\ %H:%M)] 进度 syndrome=$s pathogenesis=$p，(重)启动"
  python3 -u "$DIR/code/eval_zeroshot_local.py" --tag qwen4b_zero_run1 --outdir "$DIR/results" >> "$DIR/logs/qwen4b_zero_run1.log" 2>&1
  echo "[$(date +%m-%d\ %H:%M)] 进程退出（码 $?），5 秒后重启"
  sleep 5
done
