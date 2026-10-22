# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Find payment-scam complaints and export them for extraction
# MAGIC Part 2 of the project. Selects complaints about money transfers, bank accounts and prepaid
# MAGIC cards that mention fraud or scams (by the CFPB issue field or by keywords), then exports:
# MAGIC * `scam_extract_input.csv`: the complaints the LLM will extract facts from.
# MAGIC * `scam_label_sheet.csv`: the first rows of that same set, with empty columns for you to
# MAGIC   label by hand. Your labels are how the LLM's accuracy gets measured.
# MAGIC
# MAGIC The keyword match is deliberately broad. The LLM's `fraud_type` field then separates real
# MAGIC scams from complaints that only mention the word, and that share is a finding in itself.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(".."))

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "cfpb")
dbutils.widgets.text("extract_rows", "3000")
dbutils.widgets.text("label_rows", "120")
dbutils.widgets.text("llm_max_chars", "4000")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/raw_files"
EXTRACT_ROWS = int(dbutils.widgets.get("extract_rows"))
LABEL_ROWS = int(dbutils.widgets.get("label_rows"))
LLM_MAX_CHARS = int(dbutils.widgets.get("llm_max_chars"))
SEED = 42


def T(name):
    return f"{CATALOG}.{SCHEMA}.{name}"

# COMMAND ----------

from pyspark.sql import functions as F

from complaint_kit import scam
from complaint_kit.governance import comment_statements

silver = spark.table(T("silver_complaints"))
by_issue = F.coalesce(F.lower(F.col("issue")).rlike("fraud|scam"), F.lit(False))
by_keyword = F.coalesce(F.col("narrative_clean").rlike(scam.SCAM_KEYWORDS), F.lit(False))

candidates = (silver
              .filter(F.col("has_narrative") & F.col("product_label").isin(scam.SCAM_PRODUCTS))
              .filter(by_issue | by_keyword)
              .withColumn("matched_by", F.when(by_issue & by_keyword, "issue_and_keyword")
                          .when(by_issue, "issue").otherwise("keyword"))
              .select("complaint_id", "date_received", F.trunc("date_received", "month").alias("month"),
                      "product_label", "issue", "company_response_to_consumer", "matched_by",
                      "narrative_clean"))
candidates.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(T("gold_scam_candidates"))

cands = spark.table(T("gold_scam_candidates"))
total = cands.count()
print(f"{total:,} candidate complaints")
display(cands.groupBy("product_label", "matched_by").count().orderBy(F.desc("count")))

for stmt in comment_statements(T("gold_scam_candidates"), "gold_scam_candidates", cands.columns):
    spark.sql(stmt)

# COMMAND ----------

# MAGIC %md ## Export the extraction set and the labeling sheet

# COMMAND ----------

from complaint_kit.text import truncate

frac = min(1.0, 1.2 * EXTRACT_ROWS / total) if total else 1.0
sample = (cands.sample(fraction=frac, seed=SEED).limit(EXTRACT_ROWS).toPandas()
          .sample(frac=1, random_state=SEED).reset_index(drop=True))
sample["text"] = sample["narrative_clean"].map(lambda t: truncate(t, LLM_MAX_CHARS))

export_dir = f"{VOLUME_ROOT}/exports"
os.makedirs(export_dir, exist_ok=True)
sample[["complaint_id", "text"]].to_csv(f"{export_dir}/scam_extract_input.csv", index=False)

# The label sheet is the first LABEL_ROWS of the (shuffled) extraction set, so every
# labeled complaint also gets a model answer to compare against.
sheet = sample.head(LABEL_ROWS)[["complaint_id", "text"]].copy()
for column in list(scam.FIELDS) + [scam.AMOUNT_FIELD, "notes"]:
    sheet[column] = ""
sheet.to_csv(f"{export_dir}/scam_label_sheet.csv", index=False)

print(f"Wrote {len(sample):,} complaints to scam_extract_input.csv and "
      f"{len(sheet)} rows to scam_label_sheet.csv in {export_dir}")
print("Download both. Label the sheet (see docs/labeling_guide.md) BEFORE looking at model output.")
