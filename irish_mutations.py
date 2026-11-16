"""Shared logic for normalizing/denormalizing Irish initial mutations.

Irish word-initial consonants (and vowels) change spelling in certain grammatical
contexts:

* Lenition (séimhiú) inserts an "h" after the initial consonant:
    cat -> chat, bád -> bhád, Seán -> Sheán
* Eclipsis (urú) prefixes a consonant (or "n-" before a vowel) onto the initial
  letter, which keeps its original letter/case after the prefix:
    cat -> gcat, bád -> mbád, Cat -> gCat, Éire -> nÉire / n-Éire

`normalize_text` detects those patterns and rewrites them as the root word
preceded by a "[LEN]" or "[ECL]" marker, e.g. "a chat" -> "a [LEN] cat".
`denormalize_text` reverses that: "a [LEN] cat" -> "a chat".

This is a heuristic, dictionary-free transformation based on spelling patterns
only -- it does not consult a lexicon, so it can't tell a genuinely mutated word
from a word that natively happens to start with the same letters (e.g. "seo").
The two directions are exact inverses of each other by construction: each branch
of `demutate_word` maps to exactly one branch of `remutate_word` and back.
"""

import re

VOWELS = "aeiouáéíóú"

# Eclipsis: prefix -> underlying initial letter it replaces.
# Ordered longest-prefix-first so "bhf" is matched before the "bh" lenition pattern.
ECLIPSIS_PREFIXES = ["bhf", "mb", "gc", "nd", "ng", "bp", "dt"]

# Reverse of ECLIPSIS_PREFIXES: the underlying initial letter -> what to prepend.
_ECLIPSIS_ADDITION = {prefix[-1]: prefix[:-1] for prefix in ECLIPSIS_PREFIXES}

# Lenition: consonant + "h" -> underlying consonant.
LENITION_CONSONANTS = set("bcdfgmpst")


def _eclipsis_marker(word: str, root: str) -> str:
    """Pick "ECL" or "ECL^" for an eclipsed word/root pair.

    Eclipsis prepends one or more letters ahead of the root's own initial
    letter (e.g. "bhf" + "uil" for "fuil", "m" + "bád" for "bád"). Ordinarily
    any capitalization lives on the root itself (a proper noun: "gCorcaigh")
    or, in an all-caps title, uniformly across the whole word (handled in
    remutate_word via `root.isupper()`). But when an *ordinary* lowercase
    word is capitalized only because it opens a sentence/line (e.g. "Bhfuil
    tú...?"), the capital lands on the discarded prefix instead -- "fuil"
    alone can't distinguish "bhfuil" from "Bhfuil". "ECL^" flags that case:
    root stays lowercase, and remutate_word capitalizes the rebuilt word's
    first letter instead of the addition.
    """
    if word[0].isupper() and root[0].islower():
        return "ECL^"
    return "ECL"


def demutate_word(word: str) -> tuple[str | None, str]:
    """Return (marker, root) for a single word. marker is "LEN", "ECL", "ECL^", or None."""
    lower = word.lower()

    # Eclipsis of a vowel: "n-" + vowel. (Irish also allows a bare "n" + capital
    # vowel with no hyphen, e.g. "nAithreacha" -- but that spelling is
    # indistinguishable from "n-Aithreacha" once demutated, so detecting both
    # would make this transformation lossy. Only the unambiguous hyphenated
    # form is normalized; the bare form is left untouched.)
    if lower.startswith("n-") and len(word) > 2 and word[2].lower() in VOWELS:
        root = word[2:]
        return _eclipsis_marker(word, root), root

    # Eclipsis consonant clusters (check "bhf" before the 2-letter "bh" lenition case).
    for prefix in ECLIPSIS_PREFIXES:
        if lower.startswith(prefix) and len(word) > len(prefix):
            root = word[len(prefix) - 1 :]
            return _eclipsis_marker(word, root), root

    # Lenition: initial consonant + "h".
    if len(word) > 2 and lower[0] in LENITION_CONSONANTS and lower[1] == "h":
        return "LEN", word[0] + word[2:]

    return None, word


def remutate_word(marker: str, root: str) -> str:
    """Inverse of demutate_word: (marker, root) -> mutated word.

    Demutation preserves the case of every retained letter but discards the
    case of the removed mutation letter(s) ("h", or an eclipsis prefix like
    "g"). Standard Irish orthography only capitalizes the mutation letter
    when the whole word is capitalized (e.g. an all-caps title card), so
    `root.isupper()` is used to decide whether to reintroduce it upper- or
    lower-case.
    """
    upper = root.isupper()

    if marker == "LEN":
        h = "H" if upper else "h"
        return root[0] + h + root[1:]

    if marker in ("ECL", "ECL^"):
        first = root[0]
        lower_first = first.lower()
        if lower_first in VOWELS:
            mutated = ("N-" if upper else "n-") + root
        else:
            addition = _ECLIPSIS_ADDITION.get(lower_first)
            if addition is None:
                raise ValueError(f"Cannot reverse ECL for root starting with {first!r}")
            mutated = (addition.upper() if upper else addition) + root
        if marker == "ECL^":
            mutated = mutated[0].upper() + mutated[1:]
        return mutated

    raise ValueError(f"Unknown marker {marker!r}")


_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚáéíóú]+(?:-[A-Za-zÁÉÍÓÚáéíóú]+)*")


def normalize_text(text: str) -> str:
    """Replace mutated words in free text with '[LEN]'/'[ECL]' + root word."""

    def repl(match: re.Match[str]) -> str:
        marker, root = demutate_word(match.group(0))
        if marker is None:
            return match.group(0)
        return f"[{marker}] {root}"

    return _WORD_RE.sub(repl, text)


_MARKER_RE = re.compile(
    r"\[(LEN|ECL\^?)\] ([A-Za-zÁÉÍÓÚáéíóú]+(?:-[A-Za-zÁÉÍÓÚáéíóú]+)*)"
)


def denormalize_text(text: str) -> str:
    """Inverse of normalize_text: replace '[LEN]'/'[ECL]' markers with mutated words."""

    def repl(match: re.Match[str]) -> str:
        marker, root = match.group(1), match.group(2)
        return remutate_word(marker, root)

    return _MARKER_RE.sub(repl, text)


def normalize_text_fused(text: str) -> str:
    """Like normalize_text, but with no space between the marker and the root,
    e.g. "a chat" -> "a [LEN]cat" instead of "a [LEN] cat".

    This exists to test how much of normalize_text's effect on a downstream
    byte-level BPE tokenizer comes from the root word keeping its normal
    leading-space context (so it tokenizes identically to every other
    occurrence of that root) versus just from the mutation being spelled out.
    """

    def repl(match: re.Match[str]) -> str:
        marker, root = demutate_word(match.group(0))
        if marker is None:
            return match.group(0)
        return f"[{marker}]{root}"

    return _WORD_RE.sub(repl, text)


_MARKER_FUSED_RE = re.compile(
    r"\[(LEN|ECL\^?)\]([A-Za-zÁÉÍÓÚáéíóú]+(?:-[A-Za-zÁÉÍÓÚáéíóú]+)*)"
)


def denormalize_text_fused(text: str) -> str:
    """Inverse of normalize_text_fused."""

    def repl(match: re.Match[str]) -> str:
        marker, root = match.group(1), match.group(2)
        return remutate_word(marker, root)

    return _MARKER_FUSED_RE.sub(repl, text)
