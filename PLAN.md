# Build plan (about four weekends)

Run each step yourself and read the code as you go. Interviewers will ask why you made each
choice, and the README's "Design choices" section is your cheat sheet.

## Weekend 1: data pipeline
- [ ] Create the GitHub repo and push this folder. Commit as you go so the history shows real work.
- [ ] Sign up for Databricks Free Edition; connect the repo as a Git folder.
- [ ] Run `00_setup`. Download the four newest CFPB exports (May–August 2026) and upload them.
- [ ] Run `01_ingest_bronze`. Check the row counts per file look sensible.
- [ ] Run `02_silver_clean`. Read the product mapping and `dq_results`. Fix any unmapped product
      names in `complaint_kit/labels.py` (add a test too). Note the PII warning rates: one sentence
      in the README.
- [ ] Run `03_gold`. Open the tables in Catalog Explorer and check the comments.

## Weekend 2: routing (part 1, a few hours)
- [ ] Run `04_baseline_model`; look at the MLflow run.
- [ ] Local setup, `pytest`, API key. Download the two export files into `data/`.
- [ ] `run_llm_eval.py --dry-run`, `--limit 50`, full run. Upload results, run `05_compare_results`.
- [ ] Skim `gold_error_examples` and write two sentences on what the models get wrong.
- [ ] Improve the prompt **on the practice set, not the evaluation set**: run `04b_prompt_practice_set`,
      download `dev_sample.csv` and `example_candidates.csv` into `data/`, and save one clear
      complaint per product from the candidates as `data/few_shot_curated.csv`.
- [ ] Compare all four variants on the practice set (about $0.60):
      `python scripts/run_llm_eval.py --eval-file data/dev_sample.csv --out results/dev_results.jsonl
      --examples-file data/few_shot_curated.csv --variants zero_shot zero_shot_rules few_shot few_shot_rules`
- [ ] Edit `ROUTING_RULES` in `complaint_kit/prompts.py` based on the practice-set mistakes and rerun
      (each edit creates a new prompt version automatically). Change one thing at a time.
- [ ] Run only the best variant once on `eval_sample.csv` with `--examples-file data/few_shot_curated.csv`,
      upload `llm_results.jsonl` again, and rerun `05_compare_results`. That is your reported result.

## Weekend 3: scam extraction (part 2)
- [ ] Run `06_scam_candidates`; download the input file and the label sheet.
- [ ] Read `docs/labeling_guide.md`, then label the 120 rows (about 2–3 hours). Don't run the
      model first: seeing its answers biases your labels.
- [ ] `run_scam_extraction.py --dry-run`, then `--limit 30`, and read those 30 outputs closely.
- [ ] Full run, then `score_scam_labels.py`. Read every row of `scam_disagreements.csv` and sort them:
      model wrong, you wrong, or genuinely ambiguous. Fix your own mistakes in the sheet and rescore.
- [ ] Optional experiment: rerun with `--thinking` and compare accuracy and cost.
- [ ] If you change the prompt because of what you saw, label about 50 **new** complaints for the
      final score. Tuning a prompt on the same examples you score it on inflates its accuracy.

## Weekend 4: analysis and polish
- [ ] Upload the extraction and accuracy files; run `07_scam_analysis`.
- [ ] Build the dashboard from `dashboard/queries.sql`; screenshot it into the README.
- [ ] Write 3–5 findings in the README, each with its number and sample size. Check that every
      finding rests on enough complaints (the dashboard queries hide groups under 10).
- [ ] Update your resume (template below) and link the project title to the GitHub repo.

## Resume entry (template)

Replace every bracket with a number from your own run. If a result is unflattering, report it:
that honesty is already a strength of your other projects.

**Payment Scam Forensics with LLMs** | Python, SQL, Databricks, MLflow, Claude API, scikit-learn
- Built a Databricks pipeline over [N] CFPB complaints with data-quality checks, a product taxonomy,
  PII masking before any external API call, and a data dictionary in Unity Catalog
- Used Claude to extract scam type, contact channel, payment method, loss and refund outcome from
  [N] payment-fraud complaints at $[X] per 1,000; validated on [120] hand-labeled complaints
  ([X]%–[X]% accuracy by field, with 95% CIs)
- Found [your strongest finding, e.g. refund denial rates for tricked vs. unauthorized payments,
  or median loss by payment method, with n]
- Benchmarked LLM complaint routing against a TF-IDF baseline ([X] vs. [X] macro-F1) on the newest
  two months of held-out complaints
