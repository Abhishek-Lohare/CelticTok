"""Learn an exception list (stop-list) of words that look mutated but aren't,
from the UD train split only, then measure the heuristic with and without it
on the unseen test split.

Usage: python ud_stoplist.py <gaeilge_renyi_test folder> <UD_Irish-IDT folder>
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path
prev_expr, ud = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(prev_expr))
from irish_mutations import demutate_word, _WORD_RE

def read(split):
    for line in open(ud / f"ga_idt-ud-{split}.conllu", encoding="utf-8"):
        if not line.strip() or line.startswith("#"): continue
        c = line.rstrip("\n").split("\t")
        if "-" in c[0] or "." in c[0] or not _WORD_RE.fullmatch(c[1]): continue
        form = set()
        for kv in ([] if c[5] == "_" else c[5].split("|")):
            k, v = kv.split("=", 1)
            if k == "Form": form |= set(v.split(","))
        gold = "LEN" if "Len" in form else "ECL" if "Ecl" in form else "NONE"
        yield c[1], gold

def pred(word, stop):
    if word.lower() in stop: return "NONE"
    m, _ = demutate_word(word)
    return {"LEN": "LEN", "ECL": "ECL", "ECL^": "ECL"}.get(m, "NONE")

# 1. learn: words the heuristic flags in train that are (almost) never truly mutated
flag = defaultdict(Counter)
for w, g in read("train"):
    if pred(w, set()) != "NONE":
        flag[w.lower()][g] += 1
stop = {w for w, c in flag.items()
        if sum(c.values()) >= 3 and c["NONE"] / sum(c.values()) >= 0.9}
print(f"Stop-list learned from train ({len(stop)} words):\n  {', '.join(sorted(stop))}\n")

# 2. evaluate on test, with and without the list
for name, s in [("From Previous Experiment", set()), ("+ stop-list", stop)]:
    tp = Counter(); fp = Counter(); fn = Counter()
    for w, g in read("test"):
        p = pred(w, s)
        if p == g and g != "NONE": tp[g] += 1
        else:
            if p != "NONE": fp[p] += 1
            if g != "NONE": fn[g] += 1
    line = []
    for c in ["LEN", "ECL"]:
        P = tp[c] / (tp[c] + fp[c]); R = tp[c] / (tp[c] + fn[c])
        line.append(f"{c} P={P:.3f} R={R:.3f} F1={2*P*R/(P+R):.3f}")
    print(f"{name:<18} (test split)  " + "   ".join(line))
