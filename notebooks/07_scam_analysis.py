# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Scam analysis
# MAGIC Upload `results/scam_extractions.jsonl` (and, once you've scored your labels,
# MAGIC `results/scam_field_accuracy.json`) to the volume's `exports` folder, then run this.
# MAGIC
# MAGIC Builds `gold_scam_facts` (one row per complaint) and the summary tables behind the dashboard.
# MAGIC
# MAGIC **On purpose, nothing here ranks individual banks or companies.** Complaint counts depend
# MAGIC on customer base size and on who chooses to publish a narrative, so company rankings
# MAGIC would mislead. The questions are industry-wide: which scams, which payment methods, how
# MAGIC much is lost, and how often victims get money back.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(".."))

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "cfpb")
dbutils.widgets.text("prompt_version", "", "prompt_version (blank = latest run)")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/raw_files"
PROMPT_VERSION = dbutils.widgets.get("prompt_version").strip()


def T(name):
    return f"{CATALOG}.{SCHEMA}.{name}"

# COMMAND ----------

import json

import pandas as pd

from complaint_kit import scam
from complaint_kit.governance import comment_statements

ext = pd.read_json(f"{VOLUME_ROOT}/exports/scam_extractions.jsonl", lines=True,
                   dtype={"complaint_id": str})
if not PROMPT_VERSION:
    PROMPT_VERSION = ext.sort_values("run_at")["prompt_version"].iloc[-1]
ext = ext[ext["prompt_version"] == PROMPT_VERSION]
print(f"Using {PROMPT_VERSION}: {len(ext):,} complaints")

keep = ["complaint_id", "prompt_version", "model", *scam.FIELDS, scam.AMOUNT_FIELD,
        "amount_ok", "consistent", "cost_usd"]
ext = ext[keep].astype({"cost_usd": "float64", scam.AMOUNT_FIELD: "float64"})
spark.createDataFrame(ext).createOrReplaceTempView("ext")

facts = spark.sql(f"""
    SELECT c.complaint_id, c.date_received, c.month, c.product_label, c.issue,
           c.company_response_to_consumer, c.matched_by,
           e.fraud_type, e.scam_type, e.contact_channel, e.payment_method, e.refund_status,
           e.{scam.AMOUNT_FIELD}, e.consistent, e.prompt_version, e.model
    FROM {T('gold_scam_candidates')} c
    JOIN ext e ON CAST(c.complaint_id AS STRING) = e.complaint_id
""")
facts.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("gold_scam_facts"))
for stmt in comment_statements(T("gold_scam_facts"), "gold_scam_facts",
                               spark.table(T("gold_scam_facts")).columns):
    spark.sql(stmt)
display(spark.sql(f"SELECT fraud_type, COUNT(*) n FROM {T('gold_scam_facts')} GROUP BY 1 ORDER BY n DESC"))

# COMMAND ----------

# MAGIC %md ## Summary tables

# COMMAND ----------

F_ = T("gold_scam_facts")
summaries = {
    # What share of keyword/issue matches are actually fraud, by type, per month.
    "gold_scam_mix_by_month": f"""
        SELECT month, fraud_type, COUNT(*) AS n_complaints
        FROM {F_} GROUP BY month, fraud_type""",
    # Which scam stories are growing.
    "gold_scam_types_by_month": f"""
        SELECT month, scam_type, COUNT(*) AS n_complaints
        FROM {F_} WHERE fraud_type = 'authorized_scam' GROUP BY month, scam_type""",
    # Losses by payment method (self-reported amounts; medians resist outliers).
    "gold_scam_losses_by_method": f"""
        SELECT fraud_type, payment_method, COUNT(*) AS n_complaints,
               COUNT({scam.AMOUNT_FIELD}) AS n_with_amount,
               percentile_approx({scam.AMOUNT_FIELD}, 0.5) AS median_loss_usd,
               percentile_approx({scam.AMOUNT_FIELD}, 0.9) AS p90_loss_usd,
               SUM({scam.AMOUNT_FIELD}) AS total_reported_loss_usd
        FROM {F_} WHERE fraud_type IN ('authorized_scam', 'unauthorized_transaction') AND payment_method <> 'PARSE_ERROR'
        GROUP BY fraud_type, payment_method""",
    # Refund outcomes before the complaint, among complaints that mention one.
    "gold_scam_refunds": f"""
        SELECT fraud_type, CASE WHEN fraud_type = 'authorized_scam' THEN scam_type ELSE 'all' END AS scam_type,
               COUNT(*) AS n_mentioning_refund,
               AVG(CASE WHEN refund_status = 'denied' THEN 1.0 ELSE 0.0 END) AS pct_denied,
               AVG(CASE WHEN refund_status IN ('refunded_full', 'refunded_partial') THEN 1.0 ELSE 0.0 END) AS pct_refunded,
               AVG(CASE WHEN refund_status = 'pending' THEN 1.0 ELSE 0.0 END) AS pct_pending
        FROM {F_}
        WHERE fraud_type IN ('authorized_scam', 'unauthorized_transaction')
          AND refund_status IN ('refunded_full', 'refunded_partial', 'denied', 'pending')
        GROUP BY 1, 2""",
    # After a denied refund, how often did the complaint end in monetary relief?
    "gold_scam_relief_after_complaint": f"""
        SELECT fraud_type, refund_status, COUNT(*) AS n_complaints,
               AVG(CASE WHEN lower(company_response_to_consumer) LIKE '%monetary relief%'
                        AND lower(company_response_to_consumer) NOT LIKE '%non-monetary%'
                        THEN 1.0 ELSE 0.0 END) AS pct_monetary_relief
        FROM {F_} WHERE fraud_type IN ('authorized_scam', 'unauthorized_transaction') GROUP BY fraud_type, refund_status""",
}
for name, sql in summaries.items():
    spark.sql(sql).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T(name))
    print("wrote", name)

display(spark.table(T("gold_scam_losses_by_method")).orderBy("fraud_type", "median_loss_usd"))
display(spark.table(T("gold_scam_refunds")).orderBy("fraud_type", "scam_type"))

# COMMAND ----------

# MAGIC %md ## Extraction accuracy (from your hand labels)

# COMMAND ----------

acc_path = f"{VOLUME_ROOT}/exports/scam_field_accuracy.json"
if os.path.exists(acc_path):
    acc = pd.DataFrame(json.load(open(acc_path)))
    acc = acc[acc["n"] > 0]
    spark.createDataFrame(acc).write.mode("overwrite").option("overwriteSchema", "true") \
        .saveAsTable(T("gold_scam_extraction_accuracy"))
    display(acc)
else:
    print("No scam_field_accuracy.json yet. Run scripts/score_scam_labels.py after labeling.")
