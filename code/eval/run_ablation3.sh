#!/bin/bash
# Table 4 消融组三轮补齐（贪心+8192，与预算对齐基线同口径）
set -u
DIR="$HOME/TCMEval-SDT显著性检验_2026-09-06"
CODE="$DIR/code"; RES="$DIR/results"; LOGS="$DIR/logs"
export EVAL_TEMPERATURE=0.0 EVAL_MAX_TOKENS=8192
export VLLM_BASE_URL="http://<VLLM_HOST>:8800/v1" VLLM_MODEL_NAME="Qwen/Qwen3-4B"

run_grp () {  # $1=ablation $2=tag
  for r in 1 2 3; do
    for p in 0 1 2 3; do
      s=$((p*50)); e=$((s+50))
      python3 -u "$CODE/eval_agent.py" --vllm --task both --start "$s" --end "$e" \
        --ablation "$1" --tag "$2_run${r}_p${p}" --outdir "$RES" > "$LOGS/$2_run${r}_p${p}.log" 2>&1 &
    done
    wait
    echo "--- $2 run$r done $(date +%H:%M) ---"
  done
}

# G0 已有 run1（qwen4bGvllm_agent_run1），只补 run2/run3
for r in 2 3; do
  for p in 0 1 2 3; do
    s=$((p*50)); e=$((s+50))
    python3 -u "$CODE/eval_agent.py" --vllm --task both --start "$s" --end "$e" \
      --tag "qwen4bGvllm_agent_run${r}_p${p}" --outdir "$RES" > "$LOGS/qwen4bGvllm_agent_run${r}_p${p}.log" 2>&1 &
  done
  wait
  echo "--- G0 run$r done $(date +%H:%M) ---"
done
run_grp no_matrix G1
run_grp no_evidence G2
run_grp no_diff_guide G3
run_grp no_skill G4
echo ABLATION3_ALL_DONE
