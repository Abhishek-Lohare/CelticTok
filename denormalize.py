"""CLI: reverse normalize_text -- turn '[LEN]'/'[ECL]' + root back into the
mutated Irish word, e.g. "a [LEN] cat" -> "a chat", "a [ECL] Cat" -> "a gCat".

See irish_mutations.py for the underlying pattern logic.
"""

import argparse
import sys

from irish_mutations import denormalize_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reverse Irish lenition/eclipsis normalization ([LEN]/[ECL] markers)."
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to denormalize. If omitted, reads from stdin.",
    )
    args = parser.parse_args()

    if args.text:
        print(denormalize_text(" ".join(args.text)))
        return

    for line in sys.stdin:
        sys.stdout.write(denormalize_text(line))


if __name__ == "__main__":
    main()
