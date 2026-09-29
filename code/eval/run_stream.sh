#!/bin/bash
# 显著性检验驱动：单后端串行跑 3 次重复，每次重复内 4 切片并行
# 用法: bash run_stream.sh <backend> <nruns> [pilot]
#   backend: ds | k3 | qw32
# 环境变量（按需提供）: DS_KEY / K3_KEY 在下方读取
set -u
BACKEND="$1"; NRUNS="${2:-3}"; PILOT="${3:-full}"
DIR="$HOME/TCMEval-SDT显著性检验_2026-09-06"
CODE="$DIR/code"; RES="$DIR/results"; LOGS="$DIR/logs"
DS_KEY="sk-YOUR_API_KEY"
K3_KEY="sk-YOUR_API_KEY"

if [ "$PILOT" = "pilot" ]; then
  SLICES="0"; SLICE_LEN=3
else
  SLICES="0 1 2 3"; SLICE_LEN=50
fi

case "$BACKEND" in
  ds)
    ZERO_ENV=(DEEPSEEK_API_KEY="$DS_KEY" EVAL_API_URL="https://api.deepseek.com/v1/chat/completions" EVAL_TEMPERATURE=0.0)
    ZERO_MODEL="deepseek-v4-flash"
    AGENT_ENV=(VLLM_BASE_URL="https://api.deepseek.com/v1" VLLM_MODEL_NAME="deepseek-v4-flash" VLLM_API_KEY="$DS_KEY" VLLM_NO_THINK=1 EVAL_MAX_TOKENS=16384)
    ;;
  k3)
    ZERO_ENV=(DEEPSEEK_API_KEY="$K3_KEY" EVAL_API_URL="https://api.kimi.com/coding/v1/chat/completions" EVAL_TEMPERATURE=1.0)
    ZERO_MODEL="k3"
    AGENT_ENV=(VLLM_BASE_URL="https://api.kimi.com/coding/v1" VLLM_MODEL_NAME="k3" VLLM_API_KEY="$K3_KEY" EVAL_MAX_TOKENS=16384 EVAL_TEMPERATURE=1.0)
    ;;
  dspro)
    ZERO_ENV=(DEEPSEEK_API_KEY="$DS_KEY" EVAL_API_URL="https://api.deepseek.com/v1/chat/completions" EVAL_TEMPERATURE=0.0)
    ZERO_MODEL="deepseek-v4-pro"
    AGENT_ENV=(VLLM_BASE_URL="https://api.deepseek.com/v1" VLLM_MODEL_NAME="deepseek-v4-pro" VLLM_API_KEY="$DS_KEY" VLLM_NO_THINK=1 EVAL_MAX_TOKENS=16384)
    ;;
  qw32)
    ZERO_ENV=(DEEPSEEK_API_KEY="dummy" EVAL_API_URL="http://<VLLM_HOST>:8800/v1/chat/completions" EVAL_TEMPERATURE=0.0)
    ZERO_MODEL="Qwen/Qwen3-32B"
    AGENT_ENV=(VLLM_BASE_URL="http://<VLLM_HOST>:8800/v1" VLLM_MODEL_NAME="Qwen/Qwen3-32B" EVAL_MAX_TOKENS=4096)
    ;;
  *) echo "unknown backend"; exit 1;;
esac

run_one () {  # $1=kind(zero|agent) $2=run_idx $3=slice_idx
  local kind="$1" r="$2" p="$3"
  local start=$((p*SLICE_LEN)); local end=$((start+SLICE_LEN))
  local tag="${BACKEND}_${kind}_run${r}_p${p}"
  if [ "$kind" = "zero" ]; then
    env "${ZERO_ENV[@]}" python3 -u "$CODE/eval_zeroshot.py" \
      --model "$ZERO_MODEL" --task both --start "$start" --end "$end" \
      --tag "$tag" --outdir "$RES" > "$LOGS/${tag}.log" 2>&1
  else
    env "${AGENT_ENV[@]}" python3 -u "$CODE/eval_agent.py" --vllm --task both \
      --start "$start" --end "$end" \
      --tag "$tag" --outdir "$RES" > "$LOGS/${tag}.log" 2>&1
  fi
}

for r in $(seq 1 "$NRUNS"); do
  for kind in zero agent; do
    echo "=== $BACKEND $kind run$r $(date +%H:%M:%S) ==="
    for p in $SLICES; do run_one "$kind" "$r" "$p" & done
    wait
    echo "--- $BACKEND $kind run$r done $(date +%H:%M:%S) ---"
  done
done
echo "STREAM $BACKEND ALL DONE"
