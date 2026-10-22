"""Part 2 (scam forensics): extract structured scam facts from complaints with Claude.

Run on your laptop after notebook 06 (needs ANTHROPIC_API_KEY):

    python scripts/run_scam_extraction.py --dry-run     # prompt + cost estimate, no API calls
    python scripts/run_scam_extraction.py --limit 30    # small test run: read the outputs!
    python scripts/run_scam_extraction.py               # full run (resumable)

Optional experiment: add --thinking to let the model reason before answering.
It writes to a separate prompt version, so you can compare accuracy and cost
of both settings with score_scam_labels.py.

Writes results/scam_extractions.jsonl, one line per complaint.
"""

import argparse
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from complaint_kit import llm, scam  # noqa: E402
from complaint_kit.batch import load_jsonl, run_jobs  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(ROOT / "data" / "scam_extract_input.csv"))
    p.add_argument("--out", default=str(ROOT / "results" / "scam_extractions.jsonl"))
    p.add_argument("--model", default=llm.DEFAULT_MODEL)
    p.add_argument("--limit", type=int, help="only the first N complaints")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--thinking", action="store_true", help="let the model think before answering")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None, client=None):
    args = parse_args(argv)
    data = pd.read_csv(args.input)
    if args.limit:
        data = data.head(args.limit)
    system = scam.build_system_prompt()
    version = scam.prompt_version(system) + ("-thinking" if args.thinking else "")
    max_tokens = 4000 if args.thinking else 300

    if args.dry_run:
        est_in = (len(system) + data["text"].str.len().mean() + 40) / 4
        cost = llm.estimate_cost(args.model, est_in * len(data), 120 * len(data))
        print(f"=== {version} ===\n{system}\n")
        print(f"~{est_in:,.0f} input tokens per complaint; estimated cost for {len(data)} "
              f"complaints (without thinking): {'unknown' if cost is None else f'${cost:.2f}'}")
        print("\nFirst user message:\n" + scam.build_user_message(data["text"].iloc[0]))
        return

    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY first (see README).")
        client = llm.make_client()

    done = {(str(r["complaint_id"]), r["prompt_version"], r["model"]) for r in load_jsonl(args.out)}
    jobs = [row for row in data.itertuples()
            if (str(row.complaint_id), version, args.model) not in done]
    print(f"{len(jobs)} calls to make ({len(done)} already done).")

    def work(row):
        res = llm.classify(client, system, scam.build_user_message(row.text), args.model,
                           max_tokens=max_tokens, thinking=args.thinking)
        if res["error"]:
            return None, res["error"]
        fields = scam.parse_reply(res["reply"])
        return {
            "complaint_id": str(row.complaint_id), "prompt_version": version, "model": args.model,
            **fields, "consistent": scam.is_consistent(fields), "reply": res["reply"],
            "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
            "cost_usd": llm.estimate_cost(args.model, res["input_tokens"], res["output_tokens"]),
            "latency_s": res["latency_s"],
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, None

    run_jobs(jobs, work, args.out, args.workers)

    rows = [r for r in load_jsonl(args.out) if r["prompt_version"] == version and r["model"] == args.model]
    if not rows:
        return
    n = len(rows)
    print(f"\n{n} complaints extracted with {version}.")
    for field in scam.FIELDS:
        print(f"{field}: " + ", ".join(f"{v} {c}" for v, c in Counter(r[field] for r in rows).most_common()))
    bad = sum(any(r[f] == scam.PARSE_ERROR for f in scam.FIELDS) or not r["amount_ok"] for r in rows)
    print(f"Rows with an invalid field: {bad} ({bad / n:.1%}); rows breaking the rules: "
          f"{sum(not r['consistent'] for r in rows)}")
    cost = sum(r["cost_usd"] or 0 for r in rows)
    print(f"Estimated cost: ${cost:.2f} total, ${1000 * cost / n:.2f} per 1,000 complaints")


if __name__ == "__main__":
    main()
