"""CLI: normalize Irish lenition/eclipsis to '[LEN]'/'[ECL]' + root word.

See irish_mutations.py for the underlying pattern logic. Use denormalize.py to
reverse this transformation.
"""

import argparse
import sys

from irish_mutations import normalize_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize Irish lenition and eclipsis in text."
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to normalize. If omitted, reads from stdin.",
    )
    args = parser.parse_args()

    if args.text:
        print(normalize_text(" ".join(args.text)))
        return

    for line in sys.stdin:
        sys.stdout.write(normalize_text(line))


if __name__ == "__main__":
    main()
