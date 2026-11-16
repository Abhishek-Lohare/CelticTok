# Usage: python strategy_grid.py "<path to gaeilge_renyi_test folder>"
# Trains byte-level BPE on raw / normalized / fused / postfix Irish at 5 vocab sizes;
# prints Shannon, Renyi(2.5), Renyi efficiency H2.5/log2|V|, and Renyi with the stray space token before markers removed.
import sys, math, re
from collections import Counter
from pathlib import Path
J = Path(sys.argv[1])
sys.path.insert(0, str(J))
from irish_mutations import demutate_word, remutate_word, _WORD_RE
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
M = ["[LEN]","[ECL]","[ECL^]"]
raw = (J/"OpenSubtitles.en-ga.ga").read_text(encoding="utf-8")
def postfix(t):
    def r(m):
        k, root = demutate_word(m.group(0))
        return m.group(0) if k is None else f"{root}[{k}]"
    return _WORD_RE.sub(r, t)
PF_RE = re.compile(r"([A-Za-zÁÉÍÓÚáéíóú]+(?:-[A-Za-zÁÉÍÓÚáéíóú]+)*)\[(LEN|ECL\^?)\]")
def depostfix(t): return PF_RE.sub(lambda m: remutate_word(m.group(2), m.group(1)), t)
pf = postfix(raw)
assert depostfix(pf) == raw, "postfix not lossless"
corp = {"raw": raw,
        "normalized": (J/"OpenSubtitles.en-ga.normalized.txt").read_text(encoding="utf-8"),
        "fused": (J/"OpenSubtitles.en-ga.normalized_fused.txt").read_text(encoding="utf-8"),
        "postfix": pf}
def H(c, a):
    T = sum(c.values()); ps = [v/T for v in c.values()]
    return -sum(p*math.log2(p) for p in ps) if a == 1 else math.log2(sum(p**a for p in ps))/(1-a)
print("cond,V_target,V_trained,used,total_tokens,H1,H2.5,eff2.5,H2.5_noStrayG")
for V in [1000,2000,4000,8000,16000]:
    for name, text in corp.items():
        tk = Tokenizer(BPE()); tk.pre_tokenizer = ByteLevel(add_prefix_space=False)
        sp = [] if name == "raw" else M
        tk.train_from_iterator(text.splitlines(), BpeTrainer(vocab_size=V, min_frequency=2, special_tokens=sp, initial_alphabet=ByteLevel.alphabet(), show_progress=False))
        c = Counter(); cg = Counter()
        for e in tk.encode_batch(text.splitlines()):
            toks = e.tokens; c.update(e.ids)
            for i,(tid,tok) in enumerate(zip(e.ids,toks)):
                if tok == "Ġ" and i+1 < len(toks) and toks[i+1] in M: continue
                cg[tid] += 1
        Vt = tk.get_vocab_size(); h = H(c,2.5)
        print(f"{name},{V},{Vt},{len(c)},{sum(c.values())},{H(c,1):.4f},{h:.4f},{h/math.log2(Vt):.4f},{H(cg,2.5):.4f}")
