# -*- coding: utf-8 -*-
"""Qwen3-4B 本地零样本基线评测（transformers/MPS，thinking 默认开启，贪心解码）
用法: python3 -u eval_zeroshot_local.py [--start S] [--end E] [--tag TAG] [--outdir DIR]
输出与 eval_zeroshot.py 同构的 jsonl/summary。
"""
import argparse
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_FILE = os.path.expanduser("~/data/Train_TCM_Data_v1.json")
MODEL_PATH = os.environ.get("TCM_MODEL_PATH", os.path.expanduser("~/Qwen3-4B"))

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
    m = re.findall(r"答案\s*[:：]\s*([A-J;；、，,\s]+)", text)
    if m:
        return set(re.findall(r"[A-J]", m[-1]))
    tail = text[-100:]
    return set(re.findall(r"[A-J]", tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--tag", default="qwen4b_zero_run1")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--task", choices=["syndrome", "pathogenesis", "both"], default="both")
    ap.add_argument("--max-new", type=int, default=8192)
    args = ap.parse_args()

    records = json.load(open(DATA_FILE, encoding="utf-8"))
    if args.start or args.end:
        records = records[args.start:(args.end or None)]
    print(f"本地零样本: {MODEL_PATH}  条数={len(records)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="mps", attn_implementation="sdpa")
    model.eval()

    tasks = ["syndrome", "pathogenesis"] if args.task == "both" else [args.task]
    os.makedirs(args.outdir, exist_ok=True)
    for tk in tasks:
        task = TASKS[tk]
        jl_path = os.path.join(args.outdir, f"评估_zero_{args.tag}_{tk}.jsonl")
        done = set()
        if os.path.exists(jl_path):
            for line in open(jl_path, encoding="utf-8"):
                try:
                    done.add(json.loads(line).get("id", ""))
                except Exception:
                    pass
        n = len(records)
        with open(jl_path, "a", encoding="utf-8") as jf:
            for i, rec in enumerate(records, 1):
                rid = rec.get("Medical Record ID", "")
                if rid in done:
                    continue
                prompt, opts = build_prompt(rec, task)
                gold = parse_gold(rec[task["answers_key"]])
                msgs = [{"role": "user", "content": prompt}]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ids = tok(text, return_tensors="pt").to("mps")
                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(**ids, max_new_tokens=args.max_new,
                                         do_sample=False, temperature=None, top_p=None, top_k=None,
                                         pad_token_id=tok.eos_token_id)
                gen = out[0][ids["input_ids"].shape[1]:]
                resp = tok.decode(gen, skip_special_tokens=True)
                elapsed = time.time() - t0
                pred = {L for L in extract_letters(resp, opts) if L in opts}
                off = official_score(pred, gold)
                jf.write(json.dumps({
                    "id": rid, "gold": sorted(gold), "pred": sorted(pred),
                    "official": round(off, 4), "f1": round(f1(pred, gold), 4),
                    "hit": bool(pred & gold), "exact": pred == gold and len(pred) > 0,
                    "elapsed_s": round(elapsed, 1), "response_tail": resp[-400:],
                }, ensure_ascii=False) + "\n")
                jf.flush()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                flag = "✓" if pred == gold else ("~" if pred & gold else "✗")
                print(f"[{i}/{n}] {rid} gold={';'.join(sorted(gold))} pred={';'.join(sorted(pred)) or '-'} {flag} "
                      f"official={off:.2f} ({elapsed:.0f}s)", flush=True)
        rows = [json.loads(l) for l in open(jl_path, encoding="utf-8") if l.strip()]
        m = len(rows) or 1
        summary = {
            "task": task["label"], "model": "Qwen3-4B-local-t0", "n": len(rows),
            "official_score_mean": round(sum(r["official"] for r in rows) / m, 4),
            "exact_match": round(sum(r["exact"] for r in rows) / m, 4),
            "hit_rate": round(sum(r["hit"] for r in rows) / m, 4),
            "macro_f1": round(sum(r["f1"] for r in rows) / m, 4),
            "avg_seconds": round(sum(r["elapsed_s"] for r in rows) / m, 1),
        }
        sp = os.path.join(args.outdir, f"评估_zero_{args.tag}_{tk}.summary.json")
        json.dump(summary, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
