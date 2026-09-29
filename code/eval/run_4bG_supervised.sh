#!/bin/bash
# 4B 本地贪心智能体守护循环：进程死亡自动断点重启
DIR="$HOME/TCMEval-SDT显著性检验_2026-09-06"
while true; do
  s=$(wc -l < "$DIR/results/评估_agent_qwen4bG_agent_run1_syndrome.jsonl" 2>/dev/null || echo 0)
  p=$(wc -l < "$DIR/results/评估_agent_qwen4bG_agent_run1_pathogenesis.jsonl" 2>/dev/null || echo 0)
  if [ "$s" -ge 200 ] && [ "$p" -ge 200 ]; then echo "4BG ALL DONE"; break; fi
  echo "[$(date +%m-%d\ %H:%M)] 进度 syndrome=$s pathogenesis=$p，(重)启动"
  EVAL_GREEDY=1 EVAL_FP16=1 EVAL_MAX_TOKENS=8192 python3 -u "$DIR/code/eval_agent.py" --task both --tag qwen4bG_agent_run1 --outdir "$DIR/results" >> "$DIR/logs/qwen4bG_agent_run1.log" 2>&1
  echo "[$(date +%m-%d\ %H:%M)] 进程退出（码 $?），10 秒后重启"
  sleep 10
done
