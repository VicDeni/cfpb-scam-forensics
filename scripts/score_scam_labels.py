"""Score the model's scam extraction against your hand labels.

    python scripts/score_scam_labels.py
    python scripts/score_scam_labels.py --labels data/scam_label_sheet.csv

Prints accuracy per field with a 95% confidence interval, and writes:
  results/scam_field_accuracy.json   (upload this for the dashboard)
  results/scam_disagreements.csv     (read these: they're your error analysis)

If you ran more than one prompt version (e.g. with --thinking), each is scored.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from complaint_kit import scam  # noqa: E402
from complaint_kit.batch import load_jsonl  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default=str(ROOT / "data" / "scam_label_sheet.csv"))
    p.add_argument("--extractions", default=str(ROOT / "results" / "scam_extractions.jsonl"))
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    labels = pd.read_csv(args.labels, dtype={"complaint_id": str})
    human_rows = labels.to_dict("records")
    extractions = load_jsonl(args.extractions)
    versions = sorted({(r["prompt_version"], r["model"]) for r in extractions})
    if not versions:
        sys.exit(f"No extractions found in {args.extractions}.")

    all_results, all_disagreements, invalid = [], [], []
    for version, model in versions:
        model_rows = {r["complaint_id"]: r for r in extractions
                      if r["prompt_version"] == version and r["model"] == model}
        results, disagreements, invalid = scam.score(human_rows, model_rows)
        cost = [r["cost_usd"] or 0 for r in model_rows.values()]
        print(f"\n=== {version} ({model}) | ${1000 * sum(cost) / len(cost):.2f} per 1,000 ===")
        for r in results:
            if r["n"]:
                print(f"{r['field']:<18} {r['accuracy']:6.1%}  "
                      f"(95% CI {r['ci_low']:.0%}-{r['ci_high']:.0%}, n={r['n']})")
        all_results += [{"prompt_version": version, "model": model, **r} for r in results]
        all_disagreements += [{"prompt_version": version, **d} for d in disagreements]

    if invalid:
        print(f"\n{len(invalid)} label cells aren't valid values (typos?). Fix these in the sheet:")
        for item in invalid[:20]:
            print(f"  complaint {item['complaint_id']}, {item['field']}: {item['value']!r}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scam_field_accuracy.json", "w") as f:
        json.dump(all_results, f, indent=2)
    text = labels.set_index("complaint_id")["text"].to_dict()
    pd.DataFrame(all_disagreements).assign(
        text=lambda d: d["complaint_id"].map(text) if len(d) else None
    ).to_csv(out / "scam_disagreements.csv", index=False)
    print(f"\nWrote {out / 'scam_field_accuracy.json'} and {out / 'scam_disagreements.csv'}")
    return all_results


if __name__ == "__main__":
    main()
