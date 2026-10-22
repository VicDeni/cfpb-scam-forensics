# Data dictionary

Generated from `complaint_kit/schema.py` (the same text is written to Unity Catalog as table and column comments).

## `bronze_complaints`

Raw CFPB complaint exports, one row per record in the source files. All columns kept as strings; only column names are normalized.

| Column | Description |
|---|---|
| `source_file` | Path of the export file the row came from (lineage). |
| `ingested_at` | Timestamp when the row was loaded. |

## `silver_complaints`

Cleaned, deduplicated complaints with typed columns, a standardized product label, PII-scan flags, and masked narrative text.

| Column | Description |
|---|---|
| `complaint_id` | CFPB complaint ID. Unique in this table. |
| `date_received` | Date the CFPB received the complaint. |
| `product` | Product as filed (CFPB's naming changed over the years). |
| `sub_product` | Sub-product as filed. |
| `issue` | Issue as filed. |
| `company` | Company the complaint was sent to. |
| `state` | Consumer's state. |
| `submitted_via` | Channel (Web, Phone, Referral, ...). |
| `company_response_to_consumer` | How the company responded. |
| `timely_response` | True if the company responded on time. |
| `product_label` | Standardized product class (see labels.py); EXCLUDED for ambiguous legacy categories. |
| `has_narrative` | True if the consumer published a written narrative. |
| `pii_email` | Narrative contained an unredacted email address. |
| `pii_ssn` | Narrative contained an SSN-like pattern. |
| `pii_card` | Narrative contained a card-number-like pattern. |
| `pii_phone` | Narrative contained a phone-number-like pattern. |
| `pii_long_number` | Narrative contained a 9-19 digit number (possible account number). |
| `narrative_clean` | Narrative with residual PII masked, CFPB XXXX redactions collapsed to [REDACTED], and whitespace normalized. This is the only text sent to models. |

## `gold_model_dataset`

Modeling table: complaints with narratives and a usable label, split by time (latest months are the test set).

| Column | Description |
|---|---|
| `complaint_id` | CFPB complaint ID. |
| `date_received` | Date received. |
| `product_label` | Target class. |
| `narrative_clean` | Masked, cleaned narrative. |
| `split` | train or test. Test is the most recent months, so the model is evaluated on complaints it could not have seen. |

## `gold_complaint_trends`

Monthly complaint counts and response timeliness by product.

| Column | Description |
|---|---|
| `month` | First day of the month. |
| `product_label` | Standardized product class. |
| `n_complaints` | Complaints received. |
| `n_with_narrative` | Complaints with a published narrative. |
| `pct_timely` | Share of complaints answered on time. |

## `dq_results`

Data-quality check results, one row per check per pipeline run.

| Column | Description |
|---|---|
| `run_at` | When the checks ran. |
| `check_name` | What was checked. |
| `severity` | error stops the pipeline; warn is reported only. |
| `passed` | Whether the check passed. |
| `observed` | Observed value. |
| `threshold` | Pass condition. |

## `gold_scam_candidates`

Complaints about money transfers, bank accounts and prepaid cards that mention fraud or scams, selected for LLM extraction (part 2).

| Column | Description |
|---|---|
| `complaint_id` | CFPB complaint ID. |
| `month` | First day of the month received. |
| `matched_by` | Why it was selected: the CFPB issue field, scam keywords, or both. |
| `narrative_clean` | Masked, cleaned narrative. |

## `gold_scam_facts`

One row per candidate complaint with the facts the LLM extracted. Values and their definitions are in docs/labeling_guide.md.

| Column | Description |
|---|---|
| `fraud_type` | authorized_scam, unauthorized_transaction, or not_fraud. |
| `scam_type` | The scammer's story (authorized scams only). |
| `contact_channel` | How the scammer first made contact (authorized scams only). |
| `payment_method` | How the money left (fraud only). |
| `refund_status` | What the bank had done about the loss when the complaint was written. |
| `amount_lost_usd` | Total loss the consumer reports, in dollars (self-reported). |
| `consistent` | False if the extracted fields break the labeling rules. |
| `prompt_version` | Prompt version that produced these values. |
| `company_response_to_consumer` | How the company closed the complaint (e.g. with monetary relief). |

## `gold_model_comparison`

Baseline vs. LLM results on the same evaluation sample.

| Column | Description |
|---|---|
| `approach` | baseline_tfidf_logreg, or the LLM prompt variant. |
| `prompt_version` | Prompt name plus a hash of its exact text. |
| `model` | Model used. |
| `n` | Complaints evaluated. |
| `accuracy` | Share classified correctly. |
| `macro_f1` | F1 averaged equally across classes. |
| `parse_error_rate` | Share of LLM replies that were not a valid label. |
| `cost_per_1k_usd` | Estimated API cost per 1,000 complaints. |
