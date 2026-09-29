# -*- coding: utf-8 -*-
"""
TCMEval-SDT 基准评测脚本（证型/病机 多选任务）

数据集：Nature 子刊 Scientific Data 发表的 TCMEval-SDT 基准
  路径：~/data/
  仅 Train_TCM_Data_v1.json（200条）带标准答案，Validation/Test 无答案，故在 Train 上评测。

任务形式：每条病例给定 10 个候选选项（A-J），答案为多选（如 "B;I"）。
评测方式：候选选项以单引号包裹注入，触发智能体的「候选×四诊依据对照矩阵」证伪协议；
  eval_multi_answer 模式允许输出多个【最终证型】行，每行映射回选项字母。

用法：
  python3 评估TCMEval-SDT.py --limit 5                 # 小规模验证
  python3 评估TCMEval-SDT.py                           # 证型任务全量 200 条
  python3 评估TCMEval-SDT.py --task pathogenesis       # 病机任务
  python3 评估TCMEval-SDT.py --task both               # 两个任务

输出：
  评估结果_tCMEval_SDt_<task>.jsonl     逐条明细（含推理原文）
  评估结果_tCMEval_SDt_<task>.summary.json  汇总指标
"""
import argparse
import json
import os
import re
import sys
import time

TCM_AGENT_DIR = os.environ.get("TCM_AGENT_DIR", os.path.expanduser(os.path.expanduser("~/tcm-agent-v4")))
sys.path.insert(0, TCM_AGENT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

from harness.config import HarnessConfig
from runtime.agent_runtime import AgentRuntime, TransformersAdapter

DATA_FILE = os.path.expanduser("~/data/Train_TCM_Data_v1.json")
MODEL_PATH = os.environ.get("TCM_MODEL_PATH", os.path.expanduser("~/Qwen3-4B"))

TASKS = {
    "syndrome": {
        "options_key": "Options of TCM Syndrome",
        "answers_key": "Answers of TCM Syndrome",
        "label": "证型",
    },
    "pathogenesis": {
        "options_key": "Options of TCM Pathogenesis",
        "answers_key": "Answers of TCM Pathogenesis",
        "label": "病机",
    },
}


def parse_options(s: str) -> dict:
    """'A:风湿内侵;B:气虚血瘀;...' -> {'A': '风湿内侵', ...}"""
    out = {}
    for part in re.split(r"[;；]", s):
        part = part.strip()
        m = re.match(r"^([A-J])\s*[:：]\s*(.+)$", part)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def parse_gold(s: str) -> set:
    return {x.strip() for x in re.split(r"[;；]", s) if x.strip()}


def build_input(clinical: str, option_texts: list, label: str) -> str:
    cands = "、".join(f"'{t}'" for t in option_texts)
    return (
        f"{clinical}\n\n"
        f"候选{label}（请用对照矩阵逐一验证，可能有多个同时成立）：{cands}\n"
        f"输出格式：先给对照矩阵，结论以【最终证型】行结尾，其余章节从略。"
    )


def norm_text(t: str) -> str:
    t = re.sub(r"[\s　]", "", t)
    t = re.sub(r"[。，,、;；：:\-—（）()\[\]【】]", "", t)
    return t


def map_to_letters(pred_texts: list, options: dict) -> tuple:
    """预测文本 -> 选项字母集合；返回 (letters, unmatched_texts)"""
    letters, unmatched = set(), []
    norm_opts = {L: norm_text(t) for L, t in options.items()}
    for p in pred_texts:
        np_ = norm_text(p)
        if not np_:
            continue
        hit = None
        for L, nt in norm_opts.items():
            if np_ == nt:
                hit = L
                break
        if not hit:
            for L, nt in norm_opts.items():
                if len(np_) >= 2 and len(nt) >= 2 and (np_ in nt or nt in np_):
                    hit = L
                    break
        if hit:
            letters.add(hit)
        else:
            unmatched.append(p)
    return letters, unmatched


CONCLUSION_PATTERNS = [
    r"【最终证型】\s*[:：]?\s*([一-龥]{2,10})",
    r"最终证型\s*[:：]?\s*([一-龥]{2,10})",
    r"【证型结论】\s*[:：]?\s*([一-龥]{2,10})",
    r"\[最终证型\]\s*[:：]\s*([一-龥]{2,10})",
    r"诊断为\s*[:：]?\s*([一-龥]{2,10})",
    r"辨证为\s*[:：]?\s*([一-龥]{2,10})",
]


def extract_pred_texts(response: str, option_texts: list = None) -> list:
    """提取全部最终证型结论（评测模式允许多个），可选地用候选选项文本过滤"""
    raw = []
    for pat in CONCLUSION_PATTERNS:
        for m in re.finditer(pat, response):
            raw.append(m.group(1).strip())
    # 清理：去掉尾部说明性成分
    cleaned = []
    for t in raw:
        t = re.split(r"[\n（(]", t)[0].strip()
        t = re.sub(r"(证|的|是|为|即|主要|为主)$", "", t).strip()
        if t:
            cleaned.append(t)
    out = []
    if option_texts:
        # 只保留能映射到候选选项的结论
        for t in cleaned:
            if any(norm_text(t) == norm_text(o) or
                   (len(norm_text(t)) >= 2 and (norm_text(t) in norm_text(o) or norm_text(o) in norm_text(t)))
                   for o in option_texts):
                out.append(t)
        # 兜底：在最后一个结论标记之后的区间内找候选选项文本
        if not out:
            idx = max(response.rfind("最终证型"), response.rfind("证型结论"),
                      response.rfind("诊断结果"), response.rfind("综合结论"))
            scope = response[idx:] if idx >= 0 else response[-300:]
            for o in option_texts:
                if o in scope:
                    out.append(o)
    else:
        out = cleaned
    # 按出现顺序去重
    seen, dedup = set(), []
    for t in out:
        nt = norm_text(t)
        if nt and nt not in seen:
            seen.add(nt)
            dedup.append(t)
    return dedup


def f1(pred: set, gold: set) -> float:
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def official_score(pred: set, gold: set) -> float:
    """TCMEval-SDT 官方评分（论文 Task2/3，evaluate.py 的 score_proportional）：
    |A∩B| / (|A| + |B\\A|)，A=正确选项集，B=模型选择集。"""
    if not gold:
        return 0.0
    return len(gold & pred) / (len(gold) + len(pred - gold))


def run_task(runtime, records, task_key: str, out_prefix: str, max_new: int = 0):
    task = TASKS[task_key]
    label = task["label"]
    jl_path = f"{out_prefix}.jsonl"
    n = len(records)

    # 断点续跑：jsonl 中已有的病例直接跳过
    done_ids = set()
    if os.path.exists(jl_path):
        with open(jl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line).get("id", ""))
                except Exception:
                    pass
    if done_ids:
        print(f"  断点续跑：跳过已完成 {len(done_ids)} 条", flush=True)

    with open(jl_path, "a", encoding="utf-8") as jf:
        done_this_run = 0
        for i, rec in enumerate(records, 1):
            rid = rec.get("Medical Record ID", "")
            if rid in done_ids:
                continue
            if max_new and done_this_run >= max_new:
                print(f"  本进程已处理 {max_new} 条，退出以释放内存（防泄漏累积）", flush=True)
                break
            options = parse_options(rec[task["options_key"]])
            gold = parse_gold(rec[task["answers_key"]])
            patient_input = build_input(rec["Clinical Data"], list(options.values()), label)

            runtime.clear_context()
            t0 = time.time()
            err = ""
            eval_max_tokens = int(os.environ.get("EVAL_MAX_TOKENS", "2048"))
            try:
                response = runtime.diagnose(patient_input, max_tokens=eval_max_tokens)
            except Exception as e:  # noqa: BLE001
                response, err = "", f"{type(e).__name__}: {e}"
            elapsed = time.time() - t0

            pred_texts = extract_pred_texts(response, list(options.values()))
            pred_letters, unmatched = map_to_letters(pred_texts, options)
            exact = pred_letters == gold and len(pred_letters) > 0
            hit = len(pred_letters & gold) > 0
            f1v = f1(pred_letters, gold)
            offv = official_score(pred_letters, gold)

            quality = runtime.get_last_reasoning_quality()
            jf.write(json.dumps({
                "id": rid,
                "gold": sorted(gold),
                "pred": sorted(pred_letters),
                "pred_texts": pred_texts,
                "unmatched": unmatched,
                "exact": exact, "hit": hit, "f1": round(f1v, 4), "official": round(offv, 4),
                "elapsed_s": round(elapsed, 1),
                "reasoning_score": quality.get("score", 0.0),
                "response_tail": response[-600:],
                "error": err,
            }, ensure_ascii=False) + "\n")
            jf.flush()
            # 每例后清理 MPS 缓存，降低长跑内存压力
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

            flag = "✓" if exact else ("~" if hit else "✗")
            print(f"[{i}/{n}] {rid} "
                  f"gold={';'.join(sorted(gold))} pred={';'.join(sorted(pred_letters)) or '-'} {flag} "
                  f"official={offv:.2f} F1={f1v:.2f} ({elapsed:.0f}s)", flush=True)
            done_this_run += 1

    # 汇总：基于 jsonl 全量明细计算（含续跑部分）
    rows = [json.loads(l) for l in open(jl_path, encoding="utf-8")
            if l.strip() and not json.loads(l).get("error")]
    m = len(rows) or 1
    summary = {
        "task": label, "n": len(rows),
        "official_score_mean": round(sum(r["official"] for r in rows) / m, 4),
        "exact_match": round(sum(r["exact"] for r in rows) / m, 4),
        "hit_rate": round(sum(r["hit"] for r in rows) / m, 4),
        "macro_f1": round(sum(r["f1"] for r in rows) / m, 4),
        "unparsed": sum(1 for r in rows if not r["pred"]),
        "avg_seconds": round(sum(r["elapsed_s"] for r in rows) / m, 1),
    }
    with open(f"{out_prefix}.summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n=== {label}任务汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    ap.add_argument("--task", choices=["syndrome", "pathogenesis", "both"], default="syndrome")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--vllm", action="store_true", help="使用 vLLM 远端推理（环境变量 VLLM_BASE_URL / VLLM_MODEL_NAME）")
    ap.add_argument("--max-per-run", type=int, default=0, help="本进程最多处理 N 条后退出（配合外层循环防内存累积，0=不限）")
    ap.add_argument("--start", type=int, default=0, help="起始病例下标（切片）")
    ap.add_argument("--end", type=int, default=0, help="结束病例下标（0=到最后）")
    ap.add_argument("--tag", default="run1", help="输出文件标记")
    ap.add_argument("--outdir", default=PROJECT_ROOT, help="输出目录")
    ap.add_argument("--ablation", default="none",
                    choices=["none", "no_matrix", "no_evidence", "no_diff_guide", "no_skill"],
                    help="消融组：去掉某个机制组件（默认 none=完整机制）")
    args = ap.parse_args()

    records = json.load(open(DATA_FILE, encoding="utf-8"))
    if args.start or args.end:
        records = records[args.start:(args.end or None)]
    if args.limit > 0:
        records = records[: args.limit]
    print(f"数据: {DATA_FILE}  评测条数: {len(records)}  任务: {args.task}")

    config = HarnessConfig()
    config.eval_multi_answer = True   # 基准多选模式
    if args.ablation != "none":
        # 注意：runtime 读取的完整知识消融开关名为 ablation_no_skill_knowledge
        flag = "no_skill_knowledge" if args.ablation == "no_skill" else args.ablation
        setattr(config, f"ablation_{flag}", True)
        print(f"消融组: {args.ablation} (config.ablation_{flag}=True)")
    if args.vllm:
        from runtime.vllm_adapter import VLLMAdapter
        config.vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8081/v1")
        config.vllm_model_name = os.environ.get("VLLM_MODEL_NAME", "")
        config.vllm_request_timeout = 180.0
        if os.environ.get("EVAL_TEMPERATURE"):
            config.model_temperature = float(os.environ["EVAL_TEMPERATURE"])  # 如 K3 仅允许 temperature=1
        adapter = VLLMAdapter(config)
        print(f"后端: vLLM @ {config.vllm_base_url}  模型: {config.vllm_model_name or '(自动检测)'}")
    else:
        config.model_name_or_path = args.model
        adapter = TransformersAdapter(config)
        print(f"后端: 本地 Transformers  模型: {args.model}")
    runtime = AgentRuntime(config, adapter)
    runtime.initialize()
    runtime.skill_loader.load()
    print("智能体就绪\n")

    tasks = ["syndrome", "pathogenesis"] if args.task == "both" else [args.task]
    os.makedirs(args.outdir, exist_ok=True)
    for tk in tasks:
        run_task(runtime, records, tk,
                 os.path.join(args.outdir, f"评估_agent_{args.tag}_{tk}"),
                 max_new=args.max_per_run)


if __name__ == "__main__":
    main()
