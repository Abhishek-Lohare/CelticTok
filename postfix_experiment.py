"""Postfix-marker experiment: put the mutation marker AFTER the root word.

    raw          a chat        ár gcat
    normalized   a [LEN] cat   ár [ECL] cat     (marker before root, with space)
    fused        a [LEN]cat    ár [ECL]cat      (marker before root, no space)
    postfix      a cat[LEN]    ár cat[ECL]      (marker after root)

Reuses code unchanged:
  * irish_mutations.demutate_word / remutate_word  -> same mutation-detection rules
  * renyi_entropy.train_bpe / token_id_counts / renyi_entropy -> same tokenizer + metric

Usage (from inside gaeilge_renyi_test folder):
    uv run python postfix_experiment.py [--vocab-size 8000] [--alpha 2.5]
"""
import argparse, math, re
from collections import Counter
from pathlib import Path

from irish_mutations import demutate_word, remutate_word, _WORD_RE
from renyi_entropy import train_bpe, token_id_counts, renyi_entropy, MUTATION_MARKERS

WORD = r"[A-Za-zÁÉÍÓÚáéíóú]+(?:-[A-Za-zÁÉÍÓÚáéíóú]+)*"
_POSTFIX_RE = re.compile(rf"({WORD})\[(LEN|ECL\^?)\]")


def normalize_text_postfix(text: str) -> str:
    """'a chat' -> 'a cat[LEN]': same detection as normalize_text, marker moved after the root."""
    def repl(m):
        marker, root = demutate_word(m.group(0))
        return m.group(0) if marker is None else f"{root}[{marker}]"
    return _WORD_RE.sub(repl, text)


def denormalize_text_postfix(text: str) -> str:
    """Exact inverse: 'a cat[LEN]' -> 'a chat'."""
    return _POSTFIX_RE.sub(lambda m: remutate_word(m.group(2), m.group(1)), text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--alpha", type=float, default=2.5)
    ap.add_argument("--out-dir", type=Path, default=Path("my_output/postfix"))
    args = ap.parse_args()
    base = Path(__file__).parent
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. build the postfix corpus from the raw Irish and check it is lossless
    raw_text = (base / "OpenSubtitles.en-ga.ga").read_text(encoding="utf-8")
    postfix_text = normalize_text_postfix(raw_text)
    assert denormalize_text_postfix(postfix_text) == raw_text, "round trip failed"
    postfix_path = args.out_dir / "OpenSubtitles.en-ga.normalized_postfix.txt"
    postfix_path.write_text(postfix_text, encoding="utf-8")
    print(f"Postfix corpus written to {postfix_path} (lossless round trip: OK)\n")

    corpora = [
        ("raw",        base / "OpenSubtitles.en-ga.ga",                   []),
        ("normalized", base / "OpenSubtitles.en-ga.normalized.txt",       MUTATION_MARKERS),
        ("fused",      base / "OpenSubtitles.en-ga.normalized_fused.txt", MUTATION_MARKERS),
        ("postfix",    postfix_path,                                      MUTATION_MARKERS),
    ]
    example = "Bhí hobbit ina chónaí i bpoll sa talamh."
    examples = {"raw": example, "normalized": None, "fused": None, "postfix": normalize_text_postfix(example)}
    from irish_mutations import normalize_text, normalize_text_fused
    examples["normalized"], examples["fused"] = normalize_text(example), normalize_text_fused(example)

    rows = []
    for name, path, special in corpora:
        tok = train_bpe(path, args.vocab_size, special)                  # trainer from the previous code.
        tok.save(str(args.out_dir / f"irish_{name}_v{args.vocab_size}.json"))
        counts = token_id_counts(tok, path)                              # counter from the previous code.
        total = sum(counts.values())
        marker_ids = {tok.token_to_id(m) for m in special} - {None}
        lone_space = tok.token_to_id("Ġ")
        no_markers = Counter({t: c for t, c in counts.items() if t not in marker_ids})
        h = renyi_entropy(counts, args.alpha)                            
        rows.append(dict(
            name=name, total=total, used=len(counts),
            markers=sum(counts[i] for i in marker_ids),
            lone_space=counts.get(lone_space, 0),
            shannon=renyi_entropy(counts, 1.0), renyi=h,
            eff=h / math.log2(tok.get_vocab_size()),
            renyi_nomark=renyi_entropy(no_markers, args.alpha),
            top=[(tok.id_to_token(t), c / total) for t, c in counts.most_common(5)],
            example=tok.encode(examples[name]).tokens))

    raw_total = rows[0]["total"]
    print(f"vocab={args.vocab_size}  alpha={args.alpha}")
    print(f"{'':<11}{'tokens':>9}{'vs raw':>8}{'markers':>9}{'lone Ġ':>8}{'Shannon':>9}"
          f"{'Renyi':>8}{'eff':>7}{'Renyi-no-markers':>18}")
    for r in rows:
        print(f"{r['name']:<11}{r['total']:>9,}{100*(r['total']/raw_total-1):>+7.1f}%{r['markers']:>9,}"
              f"{r['lone_space']:>8,}{r['shannon']:>9.4f}{r['renyi']:>8.4f}{r['eff']:>7.4f}"
              f"{r['renyi_nomark']:>18.4f}")
    print("\nTop-5 tokens (share of all tokens):")
    for r in rows:
        print(f"  {r['name']:<11}" + "  ".join(f"{t!r} {100*s:.1f}%" for t, s in r["top"]))
    print(f"\nExample: {example}")
    for r in rows:
        print(f"  {r['name']:<11}{r['example']}")


if __name__ == "__main__":
    main()
