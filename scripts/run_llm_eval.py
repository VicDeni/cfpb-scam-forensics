"""Part 1 (routing): classify the evaluation sample with Claude, zero-shot and few-shot.

Run on your laptop after notebook 04 (needs ANTHROPIC_API_KEY):

    python scripts/run_llm_eval.py --dry-run          # see prompts + cost estimate, no API calls
    python scripts/run_llm_eval.py --limit 50         # small test run
    python scripts/run_llm_eval.py                    # full run

Prompt variants: zero_shot, few_shot, and the same two with "_rules" added
(routing rules written after error analysis). Try new prompts on the practice
set from notebook 04b first, so the final score on eval_sample.csv stays fair:

    python scripts/run_llm_eval.py --eval-file data/dev_sample.csv \
        --out results/dev_results.jsonl \
        --examples-file data/few_shot_curated.csv \
        --variants zero_shot zero_shot_rules few_shot few_shot_rules

Results are appended to the --out file, one line per complaint per prompt
variant. The run is resumable: rerunning skips anything already done for the
same prompt version and model, and retries rows that errored.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from complaint_kit import evaluate, llm, prompts  # noqa: E402
from complaint_kit.batch import load_jsonl, run_jobs  # noqa: E402


VARIANTS = ["zero_shot", "few_shot", "zero_shot_rules", "few_shot_rules"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-file", default=str(ROOT / "data" / "eval_sample.csv"))
    p.add_argument("--examples-file", default=str(ROOT / "data" / "few_shot_examples.csv"))
    p.add_argument("--out", default=str(ROOT / "results" / "llm_results.jsonl"))
    p.add_argument("--variants", nargs="+", default=["zero_shot", "few_shot"],
                   choices=VARIANTS)
    p.add_argument("--model", default=llm.DEFAULT_MODEL)
    p.add_argument("--limit", type=int, help="only the first N complaints (for a test run)")
    p.add_argument("--workers", type=int, default=4, help="parallel API calls")
    p.add_argument("--dry-run", action="store_true", help="show prompts and cost estimate only")
    return p.parse_args(argv)


def build_variants(names, examples_file):
    variants = {}
    for name in names:
        examples = None
        if name.startswith("few_shot"):
            examples = pd.read_csv(examples_file).to_dict("records")
        system = prompts.build_system_prompt(examples, rules=name.endswith("_rules"))
        variants[name] = {"system": system, "version": prompts.prompt_version(name, system)}
    return variants


def per_class_report(out_path, variants, model):
    """F1 by product for each variant, plus the most common mistakes."""
    records = pd.DataFrame(load_jsonl(out_path))
    if not len(records):
        return None, None
    f1_cols, mistakes = {}, []
    for name, v in variants.items():
        r = records[(records.prompt_version == v["version"]) & (records.model == model)]
        if not len(r):
            continue
        f1_cols[name] = {row["label"]: round(row["f1"], 3)
                         for row in evaluate.per_class(r.true_label, r.predicted)}
        mistakes += [{"variant": name, **m}
                     for m in evaluate.top_confusions(r.true_label, r.predicted, k=6)]
    return pd.DataFrame(f1_cols), pd.DataFrame(mistakes)


def summarize_file(out_path, variants, model):
    records = pd.DataFrame(load_jsonl(out_path))
    rows = []
    for name, v in variants.items():
        r = records[(records.prompt_version == v["version"]) & (records.model == model)] \
            if len(records) else records
        if not len(r):
            continue
        s = evaluate.summarize(r.true_label.tolist(), r.predicted.tolist(),
                               r.input_tokens.tolist(), r.output_tokens.tolist(),
                               r.cost_usd.fillna(0).tolist())
        rows.append({"variant": name, "prompt_version": v["version"], "model": model, **s})
    return rows


def main(argv=None, client=None):
    args = parse_args(argv)
    data = pd.read_csv(args.eval_file)
    if args.limit:
        data = data.head(args.limit)
    variants = build_variants(args.variants, args.examples_file)

    if args.dry_run:
        for name, v in variants.items():
            est_in = (len(v["system"]) + data["text"].str.len().mean() + 40) / 4
            cost = llm.estimate_cost(args.model, est_in * len(data), 10 * len(data))
            print(f"\n=== {name} ({v['version']}) ===\n{v['system']}\n")
            print(f"~{est_in:,.0f} input tokens per complaint; "
                  f"estimated cost for {len(data)} complaints: "
                  f"{'unknown' if cost is None else f'${cost:.2f}'}")
        print("\nFirst user message:\n" + prompts.build_user_message(data["text"].iloc[0]))
        return []

    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY first (see README).")
        client = llm.make_client()

    done = {(str(r["complaint_id"]), r["prompt_version"], r["model"]) for r in load_jsonl(args.out)}
    jobs = [(row, name, v) for name, v in variants.items() for row in data.itertuples()
            if (str(row.complaint_id), v["version"], args.model) not in done]
    print(f"{len(jobs)} calls to make ({len(done)} already done).")

    def work(job):
        row, name, v = job
        res = llm.classify(client, v["system"], prompts.build_user_message(row.text), args.model)
        if res["error"]:
            return None, res["error"]
        return {
            "complaint_id": str(row.complaint_id), "variant": name,
            "prompt_version": v["version"], "model": args.model,
            "true_label": row.product_label, "predicted": prompts.parse_label(res["reply"]),
            "reply": res["reply"], "input_tokens": res["input_tokens"],
            "output_tokens": res["output_tokens"],
            "cost_usd": llm.estimate_cost(args.model, res["input_tokens"], res["output_tokens"]),
            "latency_s": res["latency_s"],
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, None

    run_jobs(jobs, work, args.out, args.workers)

    summary = summarize_file(args.out, variants, args.model)
    print("\n" + pd.DataFrame(summary).to_string(index=False))
    f1_table, mistakes = per_class_report(args.out, variants, args.model)
    if f1_table is not None and len(f1_table):
        print("\nF1 by product:\n" + f1_table.to_string())
        print("\nMost common mistakes:\n" + mistakes.to_string(index=False))
    with open(Path(args.out).with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    main()
