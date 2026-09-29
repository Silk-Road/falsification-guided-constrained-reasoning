# -*- coding: utf-8 -*-
"""
TCMEval-SDT 零样本基线评测脚本（对照组A：裸大模型直答，无智能体机制）

设计：复刻 TCMEval-SDT 论文的基线做法——零样本提示，模型直接选择选项。
用途：与"智能体+矩阵协议"组（B组）对照，量化因果链假设检验机制的净增益。

用法：
  DEEPSEEK_API_KEY=sk-... python3 -u 评估TCMEval-SDT_零样本基线.py [--model deepseek-v4-flash] [--task syndrome|pathogenesis|both] [--limit N]

输出：
  零样本<模型名>_<task>.jsonl / .summary.json
"""
import argparse
import json
import os
import re
import time

import requests

DATA_FILE = os.path.expanduser("~/data/Train_TCM_Data_v1.json")
import os as _os
API_URL = _os.environ.get("EVAL_API_URL", "https://api.deepseek.com/v1/chat/completions")

TASKS = {
    "syndrome": {"options_key": "Options of TCM Syndrome", "answers_key": "Answers of TCM Syndrome", "label": "证型"},
    "pathogenesis": {"options_key": "Options of TCM Pathogenesis", "answers_key": "Answers of TCM Pathogenesis", "label": "病机"},
}


def parse_options(s):
    out = {}
    for part in re.split(r"[;；]", s):
        m = re.match(r"^([A-J])\s*[:：]\s*(.+)$", part.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def parse_gold(s):
    return {x.strip() for x in re.split(r"[;；]", s) if x.strip()}


def official_score(pred, gold):
    """TCMEval-SDT 官方评分（evaluate.py score_proportional）：|A∩B|/(|A|+|B\\A|)"""
    if not gold:
        return 0.0
    return len(gold & pred) / (len(gold) + len(pred - gold))


def f1(pred, gold):
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def build_prompt(rec, task):
    label = task["label"]
    opts = parse_options(rec[task["options_key"]])
    opt_text = "\n".join(f"{k}:{v}" for k, v in opts.items())
    return (
        f"以下是一则中医病例：\n{rec['Clinical Data']}\n\n"
        f"候选{label}（可多选）：\n{opt_text}\n\n"
        f"请分析病例并选择正确的{label}选项。最后一行按格式输出：答案：X（多个用分号分隔，如 答案：B;I）"
    ), opts


def extract_letters(text, options):
    """提取答案字母：优先'答案：'行或【最终证型】行，兜底全文最后一个字母集合"""
    m = re.findall(r"(?:答案|【最终证型】)\s*[:：]?\s*([A-J;；、，,\s]+)", text)
    if m:
        letters = re.findall(r"[A-J]", m[-1])
        return set(letters)
    # 兜底：末尾100字符内的选项字母
    tail = text[-100:]
    letters = re.findall(r"[A-J]", tail)
    return set(letters)


def call_api(prompt, model, api_key, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": int(_os.environ.get("EVAL_MAX_TOKENS", "8192")),
                    "temperature": float(_os.environ.get("EVAL_TEMPERATURE", "0.0")),
                    "stream": False,
                },
                timeout=300,
            )
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            return msg.get("content", "") or ""
        except Exception as e:
            if attempt == max_retries - 1:
                return f"[ERROR] {type(e).__name__}: {e}"
            time.sleep(5 * (attempt + 1))


def run_task(records, task_key, model, api_key, out_prefix):
    task = TASKS[task_key]
    jl_path = f"{out_prefix}.jsonl"
    done_ids = set()
    if os.path.exists(jl_path):
        for line in open(jl_path, encoding="utf-8"):
            try:
                done_ids.add(json.loads(line).get("id", ""))
            except Exception:
                pass
    if done_ids:
        print(f"  断点续跑：跳过 {len(done_ids)} 条", flush=True)

    n = len(records)
    with open(jl_path, "a", encoding="utf-8") as jf:
        for i, rec in enumerate(records, 1):
            rid = rec.get("Medical Record ID", "")
            if rid in done_ids:
                continue
            prompt, opts = build_prompt(rec, task)
            if os.environ.get("EVAL_FORMAT"):
                cands = "、".join(f"'{t}'" for t in opts.values())
                prompt = (f"{rec['Clinical Data']}\n\n候选{task['label']}：{cands}\n"
                          f"请分析病例并选择正确的{task['label']}选项。"
                          f"最后一行必须且只能输出：【最终证型】：X（X 必须是候选选项字母；多选时每行一个字母，如\n【最终证型】：B\n【最终证型】：I）")
            else:
                prompt = os.environ.get("EVAL_PREFIX", "") + prompt
            gold = parse_gold(rec[task["answers_key"]])
            t0 = time.time()
            resp = call_api(prompt, model, api_key)
            elapsed = time.time() - t0
            pred = extract_letters(resp, opts)
            pred = {L for L in pred if L in opts}  # 只保留合法选项
            off = official_score(pred, gold)
            jf.write(json.dumps({
                "id": rid, "gold": sorted(gold), "pred": sorted(pred),
                "official": round(off, 4), "f1": round(f1(pred, gold), 4),
                "hit": bool(pred & gold), "exact": pred == gold and len(pred) > 0,
                "elapsed_s": round(elapsed, 1), "response_tail": resp[-400:],
            }, ensure_ascii=False) + "\n")
            jf.flush()
            flag = "✓" if pred == gold else ("~" if pred & gold else "✗")
            print(f"[{i}/{n}] {rid} gold={';'.join(sorted(gold))} pred={';'.join(sorted(pred)) or '-'} {flag} "
                  f"official={off:.2f} ({elapsed:.0f}s)", flush=True)

    rows = [json.loads(l) for l in open(jl_path, encoding="utf-8") if l.strip()]
    m = len(rows) or 1
    summary = {
        "task": task["label"], "model": model, "n": len(rows),
        "official_score_mean": round(sum(r["official"] for r in rows) / m, 4),
        "exact_match": round(sum(r["exact"] for r in rows) / m, 4),
        "hit_rate": round(sum(r["hit"] for r in rows) / m, 4),
        "macro_f1": round(sum(r["f1"] for r in rows) / m, 4),
        "avg_seconds": round(sum(r["elapsed_s"] for r in rows) / m, 1),
    }
    json.dump(summary, open(f"{out_prefix}.summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n=== {task['label']} 零样本汇总 ===\n{json.dumps(summary, ensure_ascii=False, indent=2)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=_os.environ.get("EVAL_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--task", choices=["syndrome", "pathogenesis", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("EVAL_API_KEY", "")
    if not api_key:
        raise SystemExit("请设置 DEEPSEEK_API_KEY 环境变量")

    records = json.load(open(DATA_FILE, encoding="utf-8"))
    if args.start or args.end:
        records = records[args.start:(args.end or None)]
    if args.limit > 0:
        records = records[: args.limit]
    print(f"零样本基线: model={args.model} 任务={args.task} 条数={len(records)}", flush=True)

    tasks = ["syndrome", "pathogenesis"] if args.task == "both" else [args.task]
    os.makedirs(args.outdir, exist_ok=True)
    for tk in tasks:
        run_task(records, tk, args.model, api_key,
                 os.path.join(args.outdir, f"评估_zero_{args.tag}_{tk}"))


if __name__ == "__main__":
    main()
