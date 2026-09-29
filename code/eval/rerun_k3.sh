#!/bin/bash
# 任务序列：k3 agent run1 → run2 → run3 → zero run3
set -u
DIR="$HOME/TCMEval-SDT显著性检验_2026-09-06"
CODE="$DIR/code"; RES="$DIR/results"; LOGS="$DIR/logs"
K3_KEY="sk-YOUR_API_KEY"

run_slice () {  # $1=kind $2=run $3=slice
  local kind="$1" r="$2" p="$3"
  local start=$((p*50)); local end=$((start+50))
  local tag="k3_${kind}_run${r}_p${p}"
  local attempt=1
  while true; do
    local log="$LOGS/${tag}_a${attempt}.log"
    if [ "$kind" = "zero" ]; then
      DEEPSEEK_API_KEY="$K3_KEY" EVAL_API_URL="https://api.kimi.com/coding/v1/chat/completions" \
      EVAL_TEMPERATURE=1.0 python3 -u "$CODE/eval_zeroshot.py" \
        --model k3 --task both --start "$start" --end "$end" \
        --tag "$tag" --outdir "$RES" > "$log" 2>&1
    else
      VLLM_BASE_URL="https://api.kimi.com/coding/v1" VLLM_MODEL_NAME="k3" VLLM_API_KEY="$K3_KEY" \
      EVAL_MAX_TOKENS=16384 EVAL_TEMPERATURE=1.0 python3 -u "$CODE/eval_agent.py" --vllm --task both \
        --start "$start" --end "$end" --tag "$tag" --outdir "$RES" > "$log" 2>&1
    fi
    if grep -q "usage limit\|access_terminated" "$log"; then
      echo "[$(date +%H:%M:%S)] $tag 第${attempt}次尝试触发配额限制，清洗后休眠30分钟续跑"
      python3 "$CODE/purge_mock.py" "$RES" "$tag"
      attempt=$((attempt+1))
      sleep 1800
    else
      break
    fi
  done
  echo "[$(date +%H:%M:%S)] $tag 完成（${attempt}次尝试）"
}

for r in 1 2 3; do
  for p in 0 1 2 3; do run_slice agent "$r" "$p" & done
  wait
  echo "=== k3 agent run$r 完成 $(date +%H:%M:%S) ==="
done
for p in 0 1 2 3; do run_slice zero 3 "$p" & done
wait
echo "=== k3 zero run3 完成 $(date +%H:%M:%S) ==="
echo "K3 RERUN ALL DONE"
