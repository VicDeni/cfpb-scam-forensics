# Payment Scam Forensics with LLMs

What do real payment-scam complaints say about how people get scammed, how much they lose, and
whether they get their money back? This project uses an LLM to turn thousands of free-text
complaints filed with the CFPB into structured data, measures how accurate that extraction is
against hand labels, and analyzes the results. A first part benchmarks LLM complaint routing
against a classic text classifier.

Data: the CFPB Consumer Complaint Database archive (complaints received December 2011 through
August 14, 2026, with consumer-written narratives). Platform: Databricks Free Edition (Unity
Catalog, Delta tables, MLflow, AI/BI dashboard). LLM: Claude Haiku 5.5 via the Claude API.

## Questions

1. Which scam stories (bank impersonation, romance, fake sellers, fake investments...) show up
   most, and which are growing?
2. Which payment methods carry the biggest reported losses?
3. How often are refunds denied when the customer was **tricked into paying** versus when money
   was **taken without their approval**?
4. When a refund was denied, how often did complaining to the CFPB end in monetary relief?
5. How accurately can an LLM extract these facts, and at what cost?

## Findings

*Fill in from your run. Use only numbers from your own data, with their sample sizes.*

- ...
- ...
- ...

*(Dashboard screenshot here.)*

## Results

**Part 2: extraction accuracy vs. hand labels** (n = labeled complaints; 95% Wilson intervals)

| Field | Accuracy | 95% CI | n |
|---|---|---|---|
| fraud_type | | | |
| scam_type | | | |
| contact_channel | | | |
| payment_method | | | |
| refund_status | | | |
| amount_lost_usd | | | |

Extraction cost: $___ per 1,000 complaints.

**Part 1: complaint routing** (held-out complaints from the newest two months, up to 200 per product)

| Approach | Accuracy | Macro-F1 | Parse errors | Cost per 1,000 |
|---|---|---|---|---|
| TF-IDF + logistic regression | | | n/a | ~$0 |
| Claude Haiku 5.5, zero-shot | | | | |
| Claude Haiku 5.5, few-shot | | | | |

## What this project shows

| Area | What's here |
|---|---|
| Data pipeline | Bronze → silver → gold tables in Databricks, rebuilt end to end from the raw export files |
| Data quality | Checks on IDs, dates, duplicates and label coverage, logged to `dq_results`; failures stop the pipeline |
| Governance | Product taxonomy across CFPB's changing category names; PII scan and masking before any text reaches a model; data dictionary in Unity Catalog; prompt versions recorded with every result |
| LLM extraction | Structured fields from free text, with validation of every reply and rule-consistency checks |
| Evaluation | Hand-labeled sample, field-level accuracy with confidence intervals, disagreement review; baseline vs. zero-shot vs. few-shot routing on a time-based split |
| Analysis | Loss, refund and relief rates by scam type and payment method |

## Design choices

- **Tricked vs. unauthorized.** The most important field separates scams where the customer was
  tricked into sending money from transactions they never approved. Federal rules (Regulation E)
  require banks to investigate unauthorized electronic transfers, but generally don't cover payments
  the customer was tricked into making, so the two groups should have very different refund outcomes.
- **Hand labels come first.** The labeling guide and the LLM prompt are generated from the same
  definitions (`complaint_kit/scam.py`). Labels are made before looking at model output, and accuracy
  is reported with confidence intervals, because about 120 labels can't pin accuracy down to the decimal.
- **No company rankings.** Complaint counts depend on each company's size and on which customers
  publish a narrative, so ranking banks would mislead. All analysis is industry-wide.
- **Masking before the API call.** The CFPB redacts personal details, but not perfectly. Leftover
  emails, phone numbers, SSN-like and account-like numbers are masked in silver, and only masked text is
  used for modeling or sent to the LLM.
- **Invalid replies count against the model.** A field that isn't an allowed value is recorded as
  `PARSE_ERROR`, never silently fixed.
- **Time-based split for routing.** The routing test set is the newest two months, so models are scored
  on complaints from after their training data.

## How it works

```
CFPB export zips ─► bronze ─► silver ─────────────► gold_model_dataset ─► 04 baseline ─► 05 routing comparison
                     raw      typed, deduped,              │                  ▲
                              labels, PII masked,          └─► eval_sample ─► run_llm_eval.py (Claude)
                              quality checks
                                 │
                                 └─► 06 gold_scam_candidates ─► scam_extract_input ─► run_scam_extraction.py (Claude)
                                                              └► scam_label_sheet ─► you label ─► score_scam_labels.py
                                                                                                       │
                                     07 gold_scam_facts + summary tables ◄─────────────────────────────┘
                                                      └─► AI/BI dashboard
```

## Running it

### Databricks: pipeline (notebooks 00–03)
1. Sign up for [Databricks Free Edition](https://docs.databricks.com/aws/en/getting-started/free-edition).
2. Push this folder to GitHub, then in Databricks go to **Workspace → Create → Git folder** and paste the
   repo URL. (Or use **Import** to upload the folder, keeping its structure so the notebooks can import
   `complaint_kit`.)
3. Run `notebooks/00_setup`.
4. Download export zips from the
   [CFPB archive page](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/).
   Start with the four newest (May–August 2026); add older ones later for longer trends.
   Upload them in **Catalog → workspace → cfpb → raw_files → zips**.
5. Run `01_ingest_bronze`, `02_silver_clean`, `03_gold`. If `02` stops on "every product maps to a
   label", add a rule for the new product name in `complaint_kit/labels.py` (and a test), then rerun.

### Your laptop: setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                      # unit tests for the core logic
export ANTHROPIC_API_KEY=sk-ant-...   # from the Claude Console
```
Files you download from **raw_files → exports** go in `data/`; files you upload come from `results/`.

### Part 1: routing
1. Run `04_baseline_model`; download `eval_sample.csv` and `few_shot_examples.csv`.
2. `python scripts/run_llm_eval.py --dry-run`, then `--limit 50`, then the full run.
3. Upload `results/llm_results.jsonl`; run `05_compare_results`.
4. To improve the prompt, run `04b_prompt_practice_set` and iterate on `dev_sample.csv` (see PLAN.md),
   then score only the final prompt on `eval_sample.csv`.

### Part 2: scam forensics
1. Run `06_scam_candidates`; download `scam_extract_input.csv` and `scam_label_sheet.csv`.
2. Label the sheet by hand using `docs/labeling_guide.md` **before** running the model.
3. `python scripts/run_scam_extraction.py --dry-run`, then `--limit 30` (read the outputs), then the full run.
4. `python scripts/score_scam_labels.py`; read `results/scam_disagreements.csv`.
5. Upload `scam_extractions.jsonl` and `scam_field_accuracy.json`; run `07_scam_analysis`.
6. Build an AI/BI dashboard from `dashboard/queries.sql`. (If dashboards aren't available in your
   workspace, chart the same queries in a notebook or Streamlit.)

At Haiku 5.5's current price ($0.10 / $0.50 per million input / output tokens, per the
[pricing page](https://platform.claude.com/docs/en/about-claude/pricing)), both LLM steps together
should cost around $1. Each script's `--dry-run` prints an estimate.

## Repository layout

```
complaint_kit/     core logic: labels, PII/text cleaning, prompts, scam schema + scoring, Claude client, metrics
notebooks/         Databricks notebooks 00–07 (plus 04b, the prompt practice set)
scripts/           run_llm_eval.py, run_scam_extraction.py, score_scam_labels.py
dashboard/         SQL for the AI/BI dashboard
docs/              data dictionary and labeling guide (generated from complaint_kit)
tests/             pytest suite
```

## Limitations

- Only complaints whose authors agreed to publish a narrative are included, and people who complain to
  the CFPB aren't a random sample of scam victims. Findings describe complaints, not all scams.
- Loss amounts are self-reported and unverified; medians are used to limit the effect of outliers.
- Candidates are found by product, CFPB issue and keywords, so scams described without those words
  can be missed.
- Refund status reflects what the consumer wrote at the time, and may have changed later.
- Routing labels are the product the consumer chose when filing, which is sometimes wrong.
