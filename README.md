# Supplementary Code and Data

**Paper**: Falsification-Guided Constrained Reasoning: A Model-Agnostic Agent Framework for Auditable Hypothesis-Driven Decision Making

This repository contains the evaluation code, data, and per-case result logs for all experiments reported in the paper. The reasoning protocol itself is fully specified in Section 3 of the paper (matrix construction, cell judgments, aggregation and sufficiency rules, and output format) at a level that permits independent reimplementation. The agent source code and prompt templates are temporarily withheld pending a patent application; they are available from the corresponding author on reasonable request.

## Repository layout

```
code/
  eval/        TCMEval-SDT benchmark evaluation (agent & zero-shot runners,
               analysis, overlap check, mock-row cleaner, campaign drivers,
               truncation-rescue decomposition, discipline profile, 2x2
               format/content decomposition, ablation & run-level statistics,
               self-consistency baselines, L1 post-hoc budget-neutralization)
  misinfo/     misinformation-benchmark evaluation (four-condition runner,
               analysis incl. K-Acc and paired tests, unparsed-verdict sensitivity)
  figures/     figure-generation script + vector PDFs (Figures 1-5)
data/
  test_243_unique.jsonl            reconstructed 243-question misinformation set
  test_400_reconstructed.jsonl     pre-dedup reconstruction (400 rows)
results/
  tcmeval/     per-case result files (jsonl) + summaries for every reported
               configuration and repetition (TCM benchmark), incl. the C1
               (Aug-2026) zero-shot jsonl, the rescue/terminating decomposition
               output, the discipline profile / 2x2 / ablation / run-level
               statistics JSONs, and the L1 budget-neutralization outputs
  misinfo/     final (v2) per-case results for the four misinformation groups,
               incl. the unparsed-verdict sensitivity output
logs/          stdout logs of all runs, incl. the C1 agent run log
               (environment-noise warning lines and absolute paths stripped)
```

## Reproducing the headline numbers

1. **Benchmark**: TCMEval-SDT (CC-BY 4.0), only the 200-case training split has public gold answers. Place `Train_TCM_Data_v1.json` at the path referenced in `code/eval/*.py`.
2. **Zero-shot baselines**: `python3 code/eval/eval_zeroshot.py --model <model> --task both` with `EVAL_API_URL` / `DEEPSEEK_API_KEY` / `EVAL_TEMPERATURE` set (see each driver's header).
3. **Agent runs**: `python3 code/eval/eval_agent.py --vllm --task both` with `VLLM_BASE_URL` / `VLLM_MODEL_NAME` (or any OpenAI-compatible endpoint; local Qwen3-4B supported via `eval_zeroshot_local.py`).
4. **Statistics**: `python3 code/eval/analyze.py` → run-level mean±std, paired Wilcoxon, paired bootstrap 95% CI (10,000 resamples, fixed seed).
5. **Rescue/terminating decomposition** (Table `tab:decomp`): `python3 code/eval/truncation_decomposition.py` → `results/tcmeval/截断救援分解.json`.
6. **Discipline profile / 2x2 / ablation / run-level / self-consistency** (Tables `tab:discipline`, `tab:ablation`, `tab:sig`, `tab:stdbaselines`): `discipline_profile.py`, `format_content_2x2.py`, `ablation_stats.py`, `run_level_stats.py`, `self_consistency.py` (each regenerates its JSON under `results/tcmeval/`).
7. **Post-hoc budget neutralization** (Appendix F): `python3 code/eval/L1_预算中性化子样本_v2.py` (strict/typical budget-certified subsets, Table `tab:budget-neutral`), `code/eval/L1_敏感性矩阵.py` (full sensitivity matrix), `code/eval/L1_预算代理分析.py` (throughput calibration, distributions, pathology evidence), `code/eval/L1_补充校验.py` (proxy self-checks, C1 `/no_think` breakdown). Outputs land in `results/tcmeval/`.
8. **Misinformation experiment**: `python3 code/misinfo/eval_misinfo.py --group {base_original,base_misinformed,inst_misinformed,protocol_misinformed} --tag run1 --outdir results`, then `analyze_misinfo_v2.py`; the conservative unparsed-verdict sensitivity check is `敏感性分析_未解析verdict行.py`.

## Notes on campaigns and versions

Every number in the paper is tagged to a named campaign: **C1** (Aug-2026 initial runs), **C2** (Sep-2026 repetition campaign), **C3** (budget-matched greedy runs). Two API-side model drifts occurred during the campaign (a `/no_think` behavior change and a silent `deepseek-v4-flash` → v4.1 endpoint upgrade on 2026-09-19); both are documented in the paper's appendix and in the archived logs here. Runs recorded after the 2026-09-19 silent upgrade reflect a different model generation and are therefore excluded from this release (the attempted budget-symmetrized zero-shot re-run among them).

API keys are scrubbed (`sk-YOUR_API_KEY`) and internal hosts are scrubbed (`<VLLM_HOST>`); set your own via environment variables (`VLLM_BASE_URL=http://<host>:<port>/v1`).
