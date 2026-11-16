"""Evaluate spelling-based mutation detector (irish_mutations.demutate_word)
against the hand annotation in the Irish UD treebank (Form=Len / Form=Ecl).

Usage:
    python ud_eval.py <path to gaeilge_renyi_test folder> <path to UD_Irish-IDT folder>

Gold label per word: LEN if the treebank gives Form=Len, ECL if Form=Ecl, else NONE.
Predicted label: what demutate_word() returns (ECL^ counted as ECL).
Only word tokens made of letters/hyphens are scored (punctuation, numbers and
apostrophe forms like "b'" are skipped, since previous experiment's word regex skips them too).
Writes ud_eval_errors.csv with every disagreement for manual inspection.
"""
import csv, glob, sys
from collections import Counter
from pathlib import Path

Prev_expr, ud = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(Prev_expr))
from irish_mutations import demutate_word, _WORD_RE

LABELS = ["LEN", "ECL", "NONE"]
conf = Counter()            # (gold, pred) -> count
errors = Counter()          # (word, gold, pred, upos) -> count
hpref = Counter()           # h-prefixed words (a mutation previous experiment rules don't cover)
scored = skipped = 0

for f in sorted(glob.glob(str(ud / "*.conllu"))):
    for line in open(f, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        col = line.rstrip("\n").split("\t")
        if "-" in col[0] or "." in col[0]:          # multiword / empty nodes
            continue
        word, upos, feats = col[1], col[3], col[5]
        if not _WORD_RE.fullmatch(word):
            skipped += 1
            continue
        form = set()
        if feats != "_":
            for kv in feats.split("|"):
                k, v = kv.split("=", 1)
                if k == "Form":
                    form |= set(v.split(","))
        gold = "LEN" if "Len" in form else "ECL" if "Ecl" in form else "NONE"
        if "HPref" in form:
            hpref[word] += 1
        marker, _ = demutate_word(word)
        pred = {"LEN": "LEN", "ECL": "ECL", "ECL^": "ECL"}.get(marker, "NONE")
        conf[(gold, pred)] += 1
        scored += 1
        if gold != pred:
            errors[(word, gold, pred, upos)] += 1

print(f"Scored word tokens: {scored:,}   (skipped non-word tokens: {skipped:,})\n")
print("Confusion matrix (rows = treebank gold, columns = Previous heuristic)")
print(f"{'':>8}" + "".join(f"{p:>9}" for p in LABELS) + f"{'total':>9}")
for g in LABELS:
    row = [conf[(g, p)] for p in LABELS]
    print(f"{g:>8}" + "".join(f"{v:>9,}" for v in row) + f"{sum(row):>9,}")

print("\nPer-class scores")
for c in ["LEN", "ECL"]:
    tp = conf[(c, c)]
    fp = sum(conf[(g, c)] for g in LABELS if g != c)
    fn = sum(conf[(c, p)] for p in LABELS if p != c)
    P, R = tp / (tp + fp), tp / (tp + fn)
    print(f"  {c}: precision {P:.3f}  recall {R:.3f}  F1 {2*P*R/(P+R):.3f}   (TP {tp:,}, FP {fp:,}, FN {fn:,})")
tp = sum(conf[(g, p)] for g in ["LEN", "ECL"] for p in ["LEN", "ECL"])
fp = sum(conf[("NONE", p)] for p in ["LEN", "ECL"])
fn = sum(conf[(g, "NONE")] for g in ["LEN", "ECL"])
print(f"  Any mutation (ignoring type): precision {tp/(tp+fp):.3f}  recall {tp/(tp+fn):.3f}")
print(f"\nh-prefixed words in the treebank (not handled by the heuristic): {sum(hpref.values()):,}, e.g. {', '.join(w for w,_ in hpref.most_common(6))}")

def top(gold, pred, n=15):
    items = Counter()
    for (w, g, p, u), k in errors.items():
        if g == gold and p == pred:
            items[(w.lower(), u)] += k
    return ", ".join(f"{w}/{u}:{k}" for (w, u), k in items.most_common(n))
print("\nTop false positives, flagged LEN but not lenited:\n  " + top("NONE", "LEN"))
print("Top false positives, flagged ECL but not eclipsed:\n  " + top("NONE", "ECL"))
print("Top misses, lenited but not flagged:\n  " + top("LEN", "NONE"))
print("Top misses, eclipsed but not flagged:\n  " + top("ECL", "NONE"))

out = Path("ud_eval_errors.csv")
with out.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["word", "gold", "predicted", "upos", "count"])
    for (word, g, p, u), k in errors.most_common():
        w.writerow([word, g, p, u, k])
print(f"\nAll {sum(errors.values()):,} disagreements written to {out.resolve()}")
