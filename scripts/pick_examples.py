"""Hand-pick one few-shot example per product from example_candidates.csv.

    python scripts/pick_examples.py

For each product it shows the candidates (from notebook 04b) and asks for the
number of the best one. Pick a complaint that clearly belongs to its product and
that you'd label the same way without hesitating. Avoid ones that are mostly
about a different product, e.g. a credit report dispute letter as the
debt_management example. Type s to skip a product if none of its candidates are
good. Saves data/few_shot_curated.csv.
"""

import argparse
import sys
import textwrap
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["complaint_id", "label", "text"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", default=str(ROOT / "data" / "example_candidates.csv"))
    p.add_argument("--out", default=str(ROOT / "data" / "few_shot_curated.csv"))
    return p.parse_args(argv)


def main(argv=None, input_fn=input):
    args = parse_args(argv)
    candidates = pd.read_csv(args.candidates)
    picks = []
    for label, group in candidates.groupby("label", sort=True):
        rows = group.reset_index(drop=True)
        print("\n" + "=" * 78 + f"\n{label}  ({len(rows)} candidates)\n" + "=" * 78)
        for i, row in rows.iterrows():
            print(f"\n[{i + 1}] " + textwrap.fill(str(row["text"]), 96, subsequent_indent="    "))
        while True:
            answer = input_fn(f"\nBest example for {label}? 1-{len(rows)}, or s to skip: ").strip().lower()
            if answer == "s":
                break
            if answer.isdigit() and 1 <= int(answer) <= len(rows):
                picks.append(rows.iloc[int(answer) - 1][COLUMNS].to_dict())
                break
            print("Type a number from the list, or s.")
    pd.DataFrame(picks, columns=COLUMNS).to_csv(args.out, index=False)
    print(f"\nSaved {len(picks)} examples to {args.out}")
    return picks


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
