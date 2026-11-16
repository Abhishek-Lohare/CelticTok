# Does normalizing Irish initial mutations change tokenizer entropy?

This note explains the pipeline built in this repo and reports what happened
when we trained a BPE tokenizer on Irish text before and after stripping out
its grammatical spelling mutations.

## Background concepts

### Lenition and eclipsis

Irish (like the other Goidelic languages) marks a lot of grammar not with
separate words but by mutating the *first letter* of a word. There are two
mutations relevant here:

- **Lenition (séimhiú)** softens an initial consonant by inserting an `h`
  right after it: `cat` ("cat") → `chat` ("his cat"). Nine consonants can be
  lenited: `b, c, d, f, g, m, p, s, t`.
- **Eclipsis (urú)** replaces the *sound* of an initial consonant by writing
  a second consonant in front of it, while the original letter (and its
  capitalization) stays put: `cat` → `gcat` ("their cat"), `Cat` → `gCat`.
  Vowel-initial words get an `n-` prefix instead: `ainm` ("name") →
  `n-ainm` ("their name").

Both mutations are entirely predictable from grammatical context (the
preceding word, e.g. a possessive pronoun or preposition) but not from the
word alone — `chomh` could be a lenited `comh`, or it could just be the word
`chomh`. This is what makes the problem interesting for a tokenizer: the same
lexical root (`cat`, `fear`, `Meán-Domhan`, ...) surfaces under many
different spellings depending on nothing but its grammatical environment,
which inflates the vocabulary a subword tokenizer has to learn.

### Normalizing mutations away

[`irish_mutations.py`](irish_mutations.py) implements a **heuristic,
dictionary-free** normalizer:

- `normalize_text` scans each word for a mutation spelling pattern (e.g. an
  eclipsis consonant cluster, or a consonant immediately followed by `h`),
  strips it back to the presumed root, and inserts a marker token —
  `[LEN]`, `[ECL]`, or the rarer `[ECL^]` — in its place. `a chat` becomes
  `a [LEN] cat`; `a gCat` becomes `a [ECL] Cat`.
- `denormalize_text` is the exact inverse. Getting this to round-trip
  **byte-for-byte** over the full 20,035-line corpus took handling a few
  real edge cases found empirically (all-caps title cards, ambiguous
  capitalized-vowel eclipsis spellings, and sentence-initial capitals landing
  on a *discarded* prefix letter rather than the retained root — see the
  `ECL^` marker, which exists purely to carry that last bit of information
  losslessly).

Because it's pattern-based rather than dictionary-based, it will occasionally
misfire on a native word that happens to start with a mutation-shaped letter
cluster (e.g. a loanword like `seans`). That's an acceptable trade-off for
this experiment, but worth keeping in mind when reading the results below.

### Byte-Pair Encoding (BPE)

BPE is a subword tokenization algorithm: starting from individual bytes, it
repeatedly merges the most frequent adjacent pair of symbols into a new
symbol, until it reaches a target vocabulary size. The result is a vocabulary
of variable-length chunks — common whole words become single tokens, rare
words get split into smaller frequent pieces, and (with a **byte-level**
BPE, as used here) *any* input is representable without an "unknown token"
fallback, since the alphabet starts from raw bytes rather than characters.

Both tokenizers in this experiment were trained with the
[`tokenizers`](https://github.com/huggingface/tokenizers) library using
identical settings — byte-level pre-tokenizer, target vocabulary size 8000,
minimum merge frequency 2 — so that vocabulary size doesn't confound the
comparison. The only deliberate difference: the normalized corpus's trainer
was given `[LEN]`, `[ECL]`, and `[ECL^]` as **special tokens**, so they are
guaranteed to occupy a single vocabulary slot each rather than being split at
the byte level like ordinary text.

### Rényi entropy

Shannon entropy measures the average "surprise" of a distribution. Rényi
entropy is a one-parameter family that generalizes it:

$$H_\alpha(p) = \frac{1}{1-\alpha} \log_2 \left( \sum_i p_i^{\alpha} \right)$$

- As $\alpha \to 1$, $H_\alpha$ converges to ordinary Shannon entropy.
- For $\alpha > 1$ (as here, with $\alpha = 2.5$), the $p_i^\alpha$ term
  amplifies the contribution of already-frequent symbols, so $H_\alpha$
  increasingly reflects **how concentrated the distribution is on its most
  common few symbols**, and cares less about the long tail of rare ones. A
  distribution dominated by a handful of very frequent tokens gets a
  *lower* $H_{2.5}$; a flatter distribution gets a *higher* one.

Here, $p_i$ is the empirical probability of BPE token $i$ — i.e. each
tokenizer was used to re-encode its own training corpus, and $H_{2.5}$ was
computed over the resulting token-frequency distribution.

## Method

1. Train a byte-level BPE tokenizer (vocab size 8000) on
   `OpenSubtitles.en-ga.ga` (raw Irish, mutations intact).
2. Train an identical tokenizer on `OpenSubtitles.en-ga.normalized.txt` (the
   same text, run through `normalize_text`), with `[LEN]`/`[ECL]`/`[ECL^]`
   registered as special tokens.
3. Re-encode each corpus with its own tokenizer, build the empirical
   token-frequency distribution, and compute $H_1$ (Shannon) and
   $H_{2.5}$ (Rényi) over it.

Reproduce with:

```
uv run renyi_entropy.py
```

(`--vocab-size`, `--alpha`, and `--out-dir` are configurable; trained
tokenizers are saved to `tokenizer_output/`.)

## Results

| | raw Irish | normalized Irish |
|---|---:|---:|
| corpus | `OpenSubtitles.en-ga.ga` | `OpenSubtitles.en-ga.normalized.txt` |
| trained vocab size | 8000 | 8000 |
| distinct tokens actually used | 7230 | 7091 |
| total tokens over corpus | 162,524 | 188,388 |
| mutation-marker token occurrences | — | 15,299 |
| Shannon entropy ($\alpha=1$) | 9.4005 bits | 8.6674 bits |
| **Rényi entropy ($\alpha=2.5$)** | **5.7555 bits** | **5.2602 bits** |

**Raw Irish has the higher Rényi entropy, by about 0.50 bits.**

## Interpretation

Normalizing mutations away *lowers* the tokenizer's Rényi entropy at
$\alpha = 2.5$. That's the opposite of what a naive intuition ("removing
noisy spelling variants should make the corpus more diverse/informative")
might suggest — but it makes sense given what $\alpha = 2.5$ actually
measures:

- **Root words that used to fragment into multiple spellings now collapse
  onto one token.** `cat`, `chat`, `gcat`, and `Chat` used to be (partially)
  distinct byte sequences competing for vocabulary space; after
  normalization they're all just `cat`, so probability mass concentrates
  onto fewer, more frequent tokens.
- **The three marker tokens are themselves extremely frequent** — 15,299
  occurrences out of 188,388 total tokens, about 8% of the entire corpus —
  which further concentrates the distribution on a small set of high-count
  symbols.

Since $H_{2.5}$ specifically penalizes concentration on frequent symbols,
both effects push it down for the normalized corpus. In other words: this
result is a signature of the normalization doing its job (consolidating
morphological variants), not evidence that the normalized text is somehow
"simpler" or lower-information in a general sense — Shannon entropy alone
would tell a similar-shaped but less pronounced story.

**Which of those two effects actually matters more?** Recomputing
$H_{2.5}$ with the `[LEN]`/`[ECL]`/`[ECL^]` token occurrences dropped from
the distribution entirely (renormalizing over what's left) isolates the
root-consolidation effect from the markers' own frequency:

| | raw Irish | normalized | normalized, markers discarded |
|---|---:|---:|---:|
| Rényi entropy ($\alpha=2.5$) | 5.7555 bits | 5.2602 bits | 5.3163 bits |

Discarding the markers only recovers 0.056 of the 0.495-bit gap — about
**11%**. So the marker tokens' own high frequency is a minor contributor;
most of the entropy drop is really coming from mutated spellings collapsing
onto shared root tokens, which is the effect the normalizer was built to
produce in the first place. (The same holds for the fused variant: 5.2823 →
5.3405 bits with markers discarded, recovering a similarly small share.)

## Follow-up: does the space between marker and root matter?

The result above relies on the root word keeping its normal leading-space
context after normalization — `a [LEN] cat` puts a space before `cat`, so it
tokenizes to the exact same `Ġcat` byte-level token (GPT-2-style BPE encodes
"space + word" as a single unit) that plain `cat` gets everywhere else in the
corpus. That's arguably *why* normalization concentrates probability mass:
mutated spellings don't just get mapped to the same root string, they get
tokenized to the literal same token as the root elsewhere in the text.

To test that, [`irish_mutations.py`](irish_mutations.py) also has a fused
variant, `normalize_text_fused`, that glues the marker directly onto the root
with no space: `a chat` → `a [LEN]cat` instead of `a [LEN] cat` (and
`denormalize_text_fused` reverses it — confirmed byte-exact over the full
corpus, same as the spaced version). `renyi_entropy.py` trains a third
tokenizer on this fused corpus, with the markers still registered as special
tokens (so `[LEN]` stays atomic; only the space is removed):

| | normalized (`[LEN] cat`) | normalized, fused (`[LEN]cat`) |
|---|---:|---:|
| distinct tokens actually used | 7091 | 7172 |
| total tokens over corpus | 188,388 | 190,101 |
| Shannon entropy ($\alpha=1$) | 8.6674 bits | 8.7535 bits |
| **Rényi entropy ($\alpha=2.5$)** | **5.2602 bits** | **5.2823 bits** |

**Prediction confirmed in direction, but the effect is much smaller than
expected** (+0.022 bits, versus the ~0.50 bit gap between raw and normalized
Irish). Fragmentation is real and visible at the word level — e.g. `cat`
after a fused marker tokenizes as `ca` + `t` instead of the single `Ġcat`
token it gets elsewhere:

```
a [LEN]cat   -> ['a', 'Ġ', '[LEN]', 'ca', 't']
a [LEN] cat  -> ['a', 'Ġ', '[LEN]', 'Ġcat']
```

But it barely moves $H_{2.5}$, because $\alpha=2.5$ is dominated by the very
top of the frequency distribution, and that top is unchanged: `[LEN]` and
`[ECL]` are already the 3rd- and 6th-most-frequent tokens in the corpus
(11,617 and 3,680 occurrences respectively — unaffected by the spacing
choice), well ahead of any individual root fragment. `cat` splitting into
`ca`+`t` adds counts to already-common generic subword pieces rather than
creating new rare tokens, so it mostly shows up in Shannon entropy (which
weights the full tail, and did shift a bit more, +0.09 bits) rather than in
the high-alpha Rényi entropy. In short: BPE partially self-heals the missing
space by learning merges anchored to `]` instead — the tokenizer is more
resilient to this formatting choice than the "different byte sequence"
argument alone would suggest.

## Follow-up: does a smaller vocabulary budget change the story?

Everything above used a target vocab size of 8000. Rerunning with
`--vocab-size 1000` (`uv run renyi_entropy.py --vocab-size 1000`) — a much
tighter merge budget, so tokenization stays closer to raw bytes/short
fragments — gives uniformly higher entropy (a smaller vocabulary can't
concentrate mass onto a few long, specific tokens the way a bigger one can),
but the same ranking and roughly the same shape of result:

| | raw Irish | normalized | normalized, fused |
|---|---:|---:|---:|
| Rényi ($\alpha=2.5$), vocab 8000 | 5.7555 | 5.2602 | 5.2823 |
| Rényi ($\alpha=2.5$), vocab 1000 | 6.5761 | 5.9291 | 5.9550 |
| gap vs. raw (vocab 1000) | — | 0.6470 | 0.6211 |
| gap vs. raw, markers discarded (vocab 1000) | — | 0.5658 | 0.5395 |

At vocab 1000, discarding the marker tokens recovers 0.0812 of the 0.6470-bit
gap (~13%) — matching the ~11% found at vocab 8000. So the earlier
conclusion (root consolidation, not marker frequency, drives most of the
entropy drop) is not an artifact of the specific vocab-size choice; it holds
up under a 8x smaller token budget too.

## Caveats

- The normalizer is a spelling heuristic, not a lexicon-backed morphological
  analyzer, so some of the "root consolidation" effect above includes
  occasional false-positive demutations.
- Absolute entropy values depend on the chosen vocabulary size (8000) and
  corpus size (~20K subtitle lines); they're only meaningful as a
  same-settings comparison between the two corpora, not as a portable metric.
