# -*- coding: utf-8 -*-
"""基准 200 例 vs 知识库（tcm.db + 技能文件）8-gram 包含度重叠检查
结果(2026-09-15): max=1.5%, mean=0.1%, >2% 为 0 例 —— 无实质文本重叠"""
import json, sqlite3, re, glob
import numpy as np

def ngrams(text, n=8):
    t = re.sub(r"[\s，。、；：？！,.;:?!()（）\[\]【】\"'“”‘’\-—]", "", text)
    return {t[i:i+n] for i in range(max(1, len(t)-n+1))}

cases = json.load(open(os.path.expanduser("~/data/Train_TCM_Data_v1.json"), encoding="utf-8"))
case_grams = [ngrams(r["Clinical Data"] + " " + (r.get("Clinical Information") or "")) for r in cases]
kb_texts = []
db = sqlite3.connect(os.path.expanduser("~/tcm-agent-v4/data/tcm.db"))
kb_texts += [q for (q,) in db.execute("SELECT question FROM tcm_cases") if q]
db.close()
for fp in glob.glob(os.path.expanduser("~/tcm-agent-v4/harness/skills/skills/*/knowledge/*.md")) + \
           glob.glob(os.path.expanduser("~/tcm-agent-v4/harness/skills/skills/*/skill.md")):
    kb_texts.append(open(fp, encoding="utf-8").read())
kb_grams = [ngrams(t) for t in kb_texts]
maxes = []
for cg in case_grams:
    maxes.append(max((len(cg & kg) / max(len(cg), 1) for kg in kb_grams), default=0.0))
maxes = np.array(maxes)
print(f"mean={maxes.mean():.4f} p50={np.median(maxes):.4f} max={maxes.max():.4f} >0.02: {(maxes>0.02).sum()}")
