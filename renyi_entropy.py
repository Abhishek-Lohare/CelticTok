"""Train a byte-level BPE tokenizer on raw Irish vs. mutation-normalized Irish,
then compare the Renyi entropy (default alpha=2.5) of each tokenizer's token-id
distribution over its own training corpus.

The two corpora:
  * OpenSubtitles.en-ga.ga            -- raw Irish, lenition/eclipsis intact
  * OpenSubtitles.en-ga.normalized.txt -- same text with mutations rewritten as
                                          root word + a "[LEN]"/"[ECL]"/"[ECL^]"
                                          marker (see irish_mutations.py)

The normalized corpus's tokenizer is trained with those three markers as
special tokens, so they are always single vocabulary entries rather than being
split at the byte level like ordinary text.

Renyi entropy generalizes Shannon entropy:
    H_alpha(p) = 1 / (1 - alpha) * log2( sum_i p_i ** alpha )
As alpha -> 1 it converges to Shannon entropy. alpha=2.5 (> 1) weights common
tokens more heavily than rare ones, so it mostly reflects how concentrated the
distribution is on a handful of frequent subwords (a high-frequency-heavy,
"peaky" distribution has LOWER Renyi entropy than a flatter one).
"""

import argparse
import math
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

VOCAB_SIZE = 8000
MIN_FREQUENCY = 2
MUTATION_MARKERS = ["[LEN]", "[ECL]", "[ECL^]"]


def train_bpe(
    corpus_path: Path, vocab_size: int, special_tokens: list[str]
) -> Tokenizer:
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=MIN_FREQUENCY,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train([str(corpus_path)], trainer)
    return tokenizer


def token_id_counts(tokenizer: Tokenizer, corpus_path: Path) -> Counter[int]:
    lines = corpus_path.read_text(encoding="utf-8").splitlines()
    counts: Counter[int] = Counter()
    for encoding in tokenizer.encode_batch(lines):
        counts.update(encoding.ids)
    return counts


def renyi_entropy(counts: Counter[int], alpha: float) -> float:
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    if alpha == 1:
        return -sum(p * math.log2(p) for p in probs)
    return math.log2(sum(p**alpha for p in probs)) / (1 - alpha)


def analyze(
    label: str,
    corpus_path: Path,
    out_dir: Path,
    alpha: float,
    vocab_size: int,
    special_tokens: list[str],
) -> dict:
    tokenizer = train_bpe(corpus_path, vocab_size, special_tokens)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_dir / f"{label}_bpe.json"))

    counts = token_id_counts(tokenizer, corpus_path)
    marker_ids = {tokenizer.token_to_id(m) for m in special_tokens} - {None}
    marker_token_count = sum(counts[i] for i in marker_ids if i in counts)
    counts_no_markers = Counter(
        {tid: c for tid, c in counts.items() if tid not in marker_ids}
    )

    result = {
        "label": label,
        "corpus": corpus_path,
        "trained_vocab_size": tokenizer.get_vocab_size(),
        "distinct_tokens_used": len(counts),
        "total_tokens": sum(counts.values()),
        "marker_token_count": marker_token_count,
        "shannon": renyi_entropy(counts, 1.0),
        "renyi": renyi_entropy(counts, alpha),
        "renyi_no_markers": renyi_entropy(counts_no_markers, alpha),
    }

    print(f"=== {label} ===")
    print(f"  corpus: {result['corpus']}")
    print(f"  trained vocab size: {result['trained_vocab_size']}")
    print(f"  distinct tokens used: {result['distinct_tokens_used']}")
    print(f"  total tokens in corpus: {result['total_tokens']}")
    if special_tokens:
        print(f"  mutation-marker token occurrences: {result['marker_token_count']}")
    print(f"  Shannon entropy (alpha=1):    {result['shannon']:.4f} bits")
    print(f"  Renyi entropy (alpha={alpha}): {result['renyi']:.4f} bits")
    if special_tokens:
        print(
            f"  Renyi entropy (alpha={alpha}), markers discarded: "
            f"{result['renyi_no_markers']:.4f} bits"
        )
    print()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--out-dir", type=Path, default=Path("tokenizer_output"))
    args = parser.parse_args()

    base = Path(__file__).parent
    raw = analyze(
        "irish_raw",
        base / "OpenSubtitles.en-ga.ga",
        args.out_dir,
        args.alpha,
        args.vocab_size,
        special_tokens=[],
    )
    normalized = analyze(
        "irish_normalized",
        base / "OpenSubtitles.en-ga.normalized.txt",
        args.out_dir,
        args.alpha,
        args.vocab_size,
        special_tokens=MUTATION_MARKERS,
    )
    normalized_fused = analyze(
        "irish_normalized_fused",
        base / "OpenSubtitles.en-ga.normalized_fused.txt",
        args.out_dir,
        args.alpha,
        args.vocab_size,
        special_tokens=MUTATION_MARKERS,
    )

    print("=== Comparison (Renyi entropy, alpha=%.2f) ===" % args.alpha)
    results = [
        ("raw Irish", raw),
        ("normalized Irish ('[LEN] cat')", normalized),
        ("normalized Irish, fused ('[LEN]cat')", normalized_fused),
    ]
    for name, result in results:
        print(f"  {name}: {result['renyi']:.4f} bits")
    ranked = sorted(results, key=lambda pair: pair[1]["renyi"])
    print(
        f"  lowest: {ranked[0][0]} ({ranked[0][1]['renyi']:.4f}); "
        f"highest: {ranked[-1][0]} ({ranked[-1][1]['renyi']:.4f})"
    )

    print()
    print("=== With mutation-marker tokens discarded before computing entropy ===")
    for name, result in results:
        print(f"  {name}: {result['renyi_no_markers']:.4f} bits")


if __name__ == "__main__":
    main()
