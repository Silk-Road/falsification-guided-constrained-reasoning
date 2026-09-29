# -*- coding: utf-8 -*-
"""
误信息传播实验评测脚本（第九组：协议干预）
四组：base_original / base_misinformed / inst_misinformed / protocol_misinformed
用法:
  EVAL_API_KEY=sk-... python3 -u eval_misinfo.py --group base_misinformed --start 0 --end 5 --tag pilot --outdir ../results
环境变量:
  EVAL_API_URL  (默认 https://api.deepseek.com/v1/chat/completions)
  EVAL_API_KEY  (必填)
  EVAL_MODEL    (默认 deepseek-v4-flash)
  EVAL_SAMPLES  (每题采样数，默认 3)
  EVAL_TEMPERATURE (默认 0.7)
"""
import argparse
import json
import os
import re
import time

import requests

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test_243_unique.jsonl")

BASE_SYSTEM = ("You are given a question. To answer the question, you should think step by step. "
               "Use line breaks between steps, but do not use line breaks within each step. "
               "You should number each step. The final answer to the question should start with "
               "\"The answer is ...\", and should be placed at the final step. "
               "Any LaTeX expressions should be wrapped between single dollar signs, e.g., $x^2$.\n\n"
"Example output format (illustrative):\n"
"| Equation | numeric | relation | self-consistency |\n"
"| E1 | S | S | S |\n"
"| E2 | O | O | S |\n"
"| E3 | S | S | S |\n"
"【Verified】E1: $N = A \\times B$\n"
"【Corrected】E2: $X = 6 \\div P$ -> $P = 5 \\times W$ (E2 uses 6 where the question states a 5-fold relation, contradicting the given number 5)\n"
"【Verified】E3: $T = U + V$\n"
"【Usable】E1: $N = A \\times B$; E2: $P = 5 \\times W$; E3: $T = U + V$")
WARNING = "Note that the user's input could be wrong. If it has, you should point them out and correct them."
PREMISE_PREFIX = "Here are the equations that can be used to solve the problem:"

DEMO_USER_Q = ("Question: average age of students of an adult school is 40 years . 120 new students whose "
               "average age is 32 years joined the school . as a result the average age is decreased by 4 years . "
               "find the number of students of the school after joining of the new students .")
DEMO_PREMISE_OK = (PREMISE_PREFIX + " $\\text{New Average Age} = \\frac{(N_{\\text{old}} \\times A_{\\text{old}} + N_{\\text{new}} \\times A_{\\text{new}})}{N_{\\text{old}} + N_{\\text{new}}}$; \\text{New Average Age} = A_{\\text{old}} - 4; $\\text{Number of students after joining of the new students} = N_{\\text{old}} + N_{\\text{new}}$.")
DEMO_PREMISE_BAD = (PREMISE_PREFIX + " $\\text{New Average Age} = \\frac{(N_{\\text{old}} \\times A_{\\text{old}} - N_{\\text{new}} \\times A_{\\text{new}})}{N_{\\text{old}} - N_{\\text{new}}}$; $A_{\\text{old}} = \\frac{\\text{New Average Age}}{4}$; $\\text{Number of students after joining of the new students} = N_{\\text{old}} + N_{\\text{new}}$.")
DEMO_ASSISTANT_OK = ("1. The average age of students at the adult school was initially $A_{\\text{old}} = 40$ years.\n"
"2. There were $N_{\\text{new}} = 120$ new students with an average age of $A_{\\text{new}} = 32$ years.\n"
"3. After the new students joined, the average age decreased by 4 years, making $\\text{New Average Age} = A_{\\text{old}} - 4 = 36$ years.\n"
"4. Let $N_{\\text{old}}$ be the number of original students at the school. Then the total age for the original students is $40N_{\\text{old}}$.\n"
"5. The total age for the new students is $120 \\times 32 = 3840$ years.\n"
"6. The total number of students after the new students joined is $N_{\\text{old}} + 120$.\n"
"7. The total age of all students after the new students joined is $40N_{\\text{old}} + 3840$.\n"
"8. The new average age is 36 years. Using the formula for the new average age, we have $36 = \\frac{40N_{\\text{old}} + 3840}{N_{\\text{old}} + 120}$.\n"
"9. Solving the equation $36N_{\\text{old}} + 4320 = 40N_{\\text{old}} + 3840$ leads to $4N_{\\text{old}} = 480$ and hence $N_{\\text{old}} = 120$.\n"
"10. The number of students after the new students joined is $N_{\\text{old}} + N_{\\text{new}} = 120 + 120 = 240$.\n"
"11. The answer is 240.")
DEMO_CORR = ("The first formula from the user contains a mistake. It should be $\\text{New Average Age} = \\frac{(N_{\\text{old}} \\times A_{\\text{old}} + N_{\\text{new}} \\times A_{\\text{new}})}{N_{\\text{old}} + N_{\\text{new}}}$. The second formula from the user contains a mistake. It should be $\\text{New Average Age} = A_{\\text{old}} - 4$. The third formula is correct.")
DEMO_ASSISTANT_CORR = ("1. " + DEMO_CORR + "\n" + "\n".join(DEMO_ASSISTANT_OK.split("\n")[1:]).replace("1. ", "", 1) if False else
"1. " + DEMO_CORR + "\n2. The average age of students at the adult school was initially $A_{\\text{old}} = 40$ years.\n3. There were $N_{\\text{new}} = 120$ new students with an average age of $A_{\\text{new}} = 32$ years.\n4. After the new students joined, the average age decreased by 4 years, making $\\text{New Average Age} = A_{\\text{old}} - 4 = 36$ years.\n5. Let $N_{\\text{old}}$ be the number of original students at the school. Then the total age for the original students is $40N_{\\text{old}}$.\n6. The total age for the new students is $120 \\times 32 = 3840$ years.\n7. The total number of students after the new students joined is $N_{\\text{old}} + 120$.\n8. The total age of all students after the new students joined is $40N_{\\text{old}} + 3840$.\n9. The new average age is 36 years. Using the formula for the new average age, we have $36 = \\frac{40N_{\\text{old}} + 3840}{N_{\\text{old}} + 120}$.\n10. Solving the equation $36N_{\\text{old}} + 4320 = 40N_{\\text{old}} + 3840$ leads to $4N_{\\text{old}} = 480$ and hence $N_{\\text{old}} = 120$.\n11. The number of students after the new students joined is $N_{\\text{old}} + N_{\\text{new}} = 120 + 120 = 240$.\n12. The answer is 240.")

STAGE1_SYSTEM = ("You are given a question and a list of equations claimed to be useful for solving it. "
"Before any solving, verify every equation against the question.\n"
"Build a verification matrix: one row per equation (E1, E2, ...), with columns: "
"[numeric consistency (uses the question's given numbers correctly)], "
"[relation consistency (matches the relationships stated in the question)], "
"[self-consistency (does not contradict the other equations)]. "
"Each cell must be one of: S (supported), O (opposed), I (irrelevant). "
"An equation is FALSIFIED iff it has at least one O cell; the falsifying evidence must be named explicitly. "
"Then output, for each equation, exactly one line:\n"
"【Corrected】Ei: <wrong equation> -> <correct equation> (falsifying evidence)\n"
"or 【Verified】Ei: <equation>\n"
"Finally, output one line listing the equations allowed for solving:\n"
"【Usable】E1: <eq1>; E2: <eq2>; ...\n"
"Any LaTeX expressions should be wrapped between single dollar signs, e.g., $x^2$.\n\n"
"Example output format (illustrative):\n"
"| Equation | numeric | relation | self-consistency |\n"
"| E1 | S | S | S |\n"
"| E2 | O | O | S |\n"
"| E3 | S | S | S |\n"
"【Verified】E1: $N = A \\times B$\n"
"【Corrected】E2: $X = 6 \\div P$ -> $P = 5 \\times W$ (E2 uses 6 where the question states a 5-fold relation, contradicting the given number 5)\n"
"【Verified】E3: $T = U + V$\n"
"【Usable】E1: $N = A \\times B$; E2: $P = 5 \\times W$; E3: $T = U + V$")


def build_messages(rec, group):
    """返回 [(system, user), ...]；protocol 组为两段（stage1, stage2 由调用方串行）"""
    q = "Question: " + rec["question"]
    ok = PREMISE_PREFIX + " " + rec["premise"]
    bad = PREMISE_PREFIX + " " + rec["perturbed_premise"]
    if not ok.endswith("."):
        ok += "."
    if not bad.endswith("."):
        bad += "."

    def with_example(system, demo_user, demo_assistant):
        return system + "\n\nExample:\n\n" + demo_user + "\n\nAnswer:\n" + demo_assistant

    if group == "base_original":
        sys_ = with_example(BASE_SYSTEM, DEMO_PREMISE_OK + " " + DEMO_USER_Q, DEMO_ASSISTANT_OK)
        return [(sys_, ok + " " + q)]
    if group == "base_misinformed":
        sys_ = with_example(BASE_SYSTEM, DEMO_PREMISE_OK + " " + DEMO_USER_Q, DEMO_ASSISTANT_OK)
        return [(sys_, bad + " " + q)]
    if group == "inst_misinformed":
        sys_ = with_example(BASE_SYSTEM + " " + WARNING, DEMO_PREMISE_BAD + " " + DEMO_USER_Q, DEMO_ASSISTANT_CORR)
        return [(sys_, bad + " " + q)]
    if group in ("protocol_misinformed", "protocol_original"):
        return [(STAGE1_SYSTEM, (bad if group == "protocol_misinformed" else ok) + " " + q)]
    raise ValueError(group)


NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")


def extract_number(text):
    """复刻其 answer_verifier：'The answer is' 后、最后一个 '=' 后的第一个数；否则全串最后一个数"""
    m = re.findall(r"[Tt]he answer is\s*[:：]?\s*([^\n]+)", text)
    scope = m[-1] if m else text
    if "=" in scope:
        scope = scope.split("=")[-1]
    frac = re.search(r"(-?\d[\d,]*(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", scope)
    if frac:
        try:
            return float(frac.group(1).replace(",", "")) / float(frac.group(2))
        except Exception:
            pass
    nums = NUM_RE.findall(scope)
    if not nums and m:
        nums = NUM_RE.findall(text)
    if not nums:
        return None
    raw = nums[0] if m else nums[-1]
    try:
        return float(raw.replace(",", ""))
    except Exception:
        return None


def call_api(messages, model, api_key, api_url, temperature, max_tokens=4096, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "temperature": temperature,
                      "max_tokens": max_tokens, "stream": False},
                timeout=300,
            )
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            return (msg.get("content") or ""), (msg.get("reasoning_content") or "")
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                return f"[ERROR] {type(e).__name__}: {e}", ""
            time.sleep(5 * (attempt + 1))


def run_row(rec, group, model, api_key, api_url, temperature):
    """返回 dict：stage1_output(仅 protocol)、response、verdicts(仅 protocol)"""
    stages = build_messages(rec, group)
    if group not in ("protocol_misinformed", "protocol_original"):
        sys_, user = stages[0]
        content, reasoning = call_api([{"role": "system", "content": sys_}, {"role": "user", "content": user}],
                        model, api_key, api_url, temperature)
        return {"response": content, "reasoning": reasoning}
    # protocol：两段
    sys1, user1 = stages[0]
    s1, _r1 = call_api([{"role": "system", "content": sys1}, {"role": "user", "content": user1}],
                  model, api_key, api_url, temperature, max_tokens=16384)
    m = re.search(r"【Usable】\s*(.+)", s1)
    if m:
        usable = m.group(1).strip()
        eq_part = re.sub(r"^E\d+\s*[:：]\s*", "", usable)
        eq_part = re.sub(r";\s*E\d+\s*[:：]\s*", "; ", eq_part)
        usable_text = PREMISE_PREFIX + " " + eq_part
    else:
        usable_text = ("Verification report on the user-provided equations:\n" + s1 +
                       "\nUse only the equations verified or corrected above.")
    sys2 = BASE_SYSTEM + "\n\nExample:\n\n" + DEMO_PREMISE_OK + " " + DEMO_USER_Q + "\n\nAnswer:\n" + DEMO_ASSISTANT_OK
    user2 = usable_text + " Question: " + rec["question"]
    resp, reasoning = call_api([{"role": "system", "content": sys2}, {"role": "user", "content": user2}],
                    model, api_key, api_url, temperature, max_tokens=8192)
    verdicts = {}
    for mm in re.finditer(r"【(Corrected|Verified)】\s*E(\d+)", s1):
        verdicts[int(mm.group(2))] = (mm.group(1) == "Corrected")
    return {"stage1_output": s1, "response": resp, "reasoning": reasoning, "verdicts": verdicts}


def ground_truth_flags(rec):
    p = [x.strip() for x in rec["premise"].split(";") if x.strip()]
    pp = [x.strip() for x in rec["perturbed_premise"].split(";") if x.strip()]
    if len(p) != len(pp):
        return None
    return {i + 1: (a != b) for i, (a, b) in enumerate(zip(p, pp))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True,
                    choices=["base_original", "base_misinformed", "inst_misinformed", "protocol_misinformed", "protocol_original"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--outdir", default="../results")
    args = ap.parse_args()

    api_key = os.environ.get("EVAL_API_KEY", "")
    api_url = os.environ.get("EVAL_API_URL", "https://api.deepseek.com/v1/chat/completions")
    model = os.environ.get("EVAL_MODEL", "deepseek-v4-flash")
    temperature = float(os.environ.get("EVAL_TEMPERATURE", "0.7"))
    n_samples = int(os.environ.get("EVAL_SAMPLES", "3"))
    if not api_key:
        raise SystemExit("请设置 EVAL_API_KEY")

    records = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    if args.start or args.end:
        records = records[args.start:(args.end or None)]
    os.makedirs(args.outdir, exist_ok=True)
    jl_path = os.path.join(args.outdir, f"评估_misinfo_{args.tag}_{args.group}.jsonl")

    done = set()
    if os.path.exists(jl_path):
        for line in open(jl_path, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["idx"], r["sample"]))
            except Exception:
                pass
    if done:
        print(f"  断点续跑：跳过 {len(done)} 条", flush=True)

    n = len(records)
    with open(jl_path, "a", encoding="utf-8") as jf:
        for i, rec in enumerate(records, 1):
            idx = args.start + i - 1
            gold = extract_number(str(rec["correct_answer"]))
            gt = ground_truth_flags(rec)
            if args.group == "protocol_original":
                gt = ({i: False for i in gt} if gt is not None else None)
            for k in range(n_samples):
                if (idx, k) in done:
                    continue
                t0 = time.time()
                out = run_row(rec, args.group, model, api_key, api_url, temperature)
                elapsed = time.time() - t0
                resp = out.get("response", "")
                pred = extract_number(resp)
                correct = (pred is not None and gold is not None and abs(pred - gold) < 1e-6)
                row = {
                    "idx": idx, "sample": k, "group": args.group,
                    "gold": gold, "pred": pred, "correct": correct,
                    "elapsed_s": round(elapsed, 1),
                    "response_tail": resp[-400:],
                    "reasoning_tail": out.get("reasoning", "")[-200:],
                    "error": resp[:7] == "[ERROR]",
                }
                if args.group in ("protocol_misinformed", "protocol_original"):
                    row["stage1_tail"] = out.get("stage1_output", "")[-400:]
                    v = out.get("verdicts", {})
                    row["verdicts"] = {str(k2): v2 for k2, v2 in v.items()}
                    if gt is not None and v:
                        tp = sum(1 for j, f in gt.items() if f and v.get(j) is True)
                        fp = sum(1 for j, f in gt.items() if not f and v.get(j) is True)
                        fn = sum(1 for j, f in gt.items() if f and v.get(j) is not True)
                        row["detect"] = {"tp": tp, "fp": fp, "fn": fn,
                                         "n_perturbed": sum(gt.values()), "n_eq": len(gt)}
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                jf.flush()
                flag = "✓" if correct else "✗"
                print(f"[{i}/{n} k{k}] idx={idx} gold={gold} pred={pred} {flag} ({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
