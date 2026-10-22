# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Clean → silver, with data-quality checks
# MAGIC * Types the columns, removes duplicate complaints, and standardizes the product label.
# MAGIC * Scans narratives for personal information the CFPB's redaction missed and masks it.
# MAGIC   Only the masked text (`narrative_clean`) is used for modeling or sent to the LLM.
# MAGIC * Runs data-quality checks and records them in `dq_results`. An **error** check failing
# MAGIC   stops the notebook, so bad data never reaches the gold tables.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(".."))

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "cfpb")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")


def T(name):
    return f"{CATALOG}.{SCHEMA}.{name}"

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

from complaint_kit import labels, text

bronze = spark.table(T("bronze_complaints"))

OPTIONAL = ["sub_product", "sub_issue", "state", "submitted_via",
            "company_response_to_consumer", "timely_response"]
for col in OPTIONAL:
    if col not in bronze.columns:
        bronze = bronze.withColumn(col, F.lit(None).cast("string"))


def parse_date(col):
    c = F.trim(F.col(col))
    return F.to_date(F.coalesce(
        F.expr(f"try_to_timestamp(substr(trim({col}), 1, 10), 'yyyy-MM-dd')"),
        F.expr(f"try_to_timestamp(trim({col}), 'MM/dd/yyyy')"),
        F.expr(f"try_to_timestamp(trim({col}), 'MM/dd/yy')"),
    ))


def yes_no(col):
    v = F.lower(F.trim(F.col(col)))
    return F.when(v.isin("yes", "true"), True).when(v.isin("no", "false"), False)


typed = bronze.select(
    F.expr("try_cast(trim(complaint_id) AS BIGINT)").alias("complaint_id"),
    parse_date("date_received").alias("date_received"),
    F.trim("product").alias("product"),
    F.trim("sub_product").alias("sub_product"),
    F.trim("issue").alias("issue"),
    F.trim("sub_issue").alias("sub_issue"),
    F.trim("company").alias("company"),
    F.trim("state").alias("state"),
    F.trim("submitted_via").alias("submitted_via"),
    F.trim("company_response_to_consumer").alias("company_response_to_consumer"),
    yes_no("timely_response").alias("timely_response"),
    F.col("consumer_complaint_narrative").alias("narrative_raw"),
    "source_file",
)

# Keep one row per complaint (the most recent export wins).
w = Window.partitionBy("complaint_id").orderBy(F.col("source_file").desc())
deduped = (typed.filter(F.col("complaint_id").isNotNull())
           .withColumn("_rn", F.row_number().over(w))
           .filter("_rn = 1").drop("_rn"))

# COMMAND ----------

# MAGIC %md ## Standardize product labels
# MAGIC There are only a few dozen distinct product names, so map them in Python and join.

# COMMAND ----------

products = [r["product"] for r in deduped.select("product").distinct().collect()]
mapping = spark.createDataFrame(
    [(p, labels.map_product(p)) for p in products], "product string, product_label string"
)
display(mapping.orderBy("product_label", "product"))

with_labels = deduped.join(mapping, deduped["product"].eqNullSafe(mapping["product"]), "left") \
    .drop(mapping["product"])

# COMMAND ----------

# MAGIC %md ## Scan for leftover PII, mask it, clean the text
# MAGIC Same regex patterns as `complaint_kit/text.py` (unit-tested there).

# COMMAND ----------

narr = F.col("narrative_raw")
silver = with_labels.withColumn(
    "has_narrative", F.coalesce(F.length(F.trim(narr)) > 0, F.lit(False))
)
for name, pattern in text.PII_PATTERNS.items():
    silver = silver.withColumn(f"pii_{name}", F.coalesce(narr.rlike(pattern), F.lit(False)))

clean = narr
for name, pattern in text.PII_PATTERNS.items():
    clean = F.regexp_replace(clean, pattern, text.PII_TOKENS[name])
clean = F.regexp_replace(clean, text.REDACTION_PATTERN, "[REDACTED]")
clean = F.regexp_replace(clean, text.REDACTED_RUN_PATTERN, "[REDACTED] ")
clean = F.regexp_replace(clean, text.BRACES_PATTERN, "")
clean = F.trim(F.regexp_replace(clean, text.WHITESPACE_PATTERN, " "))

silver = (silver
          .withColumn("narrative_clean", F.when(F.col("has_narrative"), clean))
          .drop("narrative_raw"))  # raw text stays only in bronze

(silver.write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(T("silver_complaints")))

# COMMAND ----------

# MAGIC %md ## Data-quality checks

# COMMAND ----------

import datetime as dt

s = spark.table(T("silver_complaints"))
n_bronze = spark.table(T("bronze_complaints")).count()
n_typed_ids = typed.filter(F.col("complaint_id").isNotNull()).count()
n_silver = s.count()

stats = s.agg(
    F.sum(F.col("date_received").isNull().cast("int")).alias("null_dates"),
    F.min("date_received").alias("min_date"),
    F.max("date_received").alias("max_date"),
    F.countDistinct("complaint_id").alias("distinct_ids"),
    F.avg(F.col("has_narrative").cast("double")).alias("narrative_share"),
    *[F.avg(F.when(F.col("has_narrative"), F.col(f"pii_{k}").cast("double"))).alias(f"pii_{k}")
      for k in text.PII_PATTERNS],
).first()

unmapped = [r["product"] for r in s.filter(F.col("product_label") == labels.UNMAPPED)
            .select("product").distinct().collect()]

leftover_pii = s.filter(
    F.col("narrative_clean").rlike("|".join(f"({p})" for p in text.PII_PATTERNS.values()))
).count()

checks = [
    ("complaint_id parses for every row", "error", n_bronze - n_typed_ids == 0,
     f"{n_bronze - n_typed_ids} unparseable", "0"),
    ("complaint_id unique in silver", "error", stats["distinct_ids"] == n_silver,
     f"{n_silver - stats['distinct_ids']} duplicates", "0"),
    ("duplicate rows removed", "warn", True, f"{n_typed_ids - n_silver} removed", "reported"),
    ("date_received parses", "error", stats["null_dates"] <= 0.001 * n_silver,
     f"{stats['null_dates']} null", "<= 0.1% of rows"),
    ("dates within CFPB range", "error",
     stats["min_date"] is not None and stats["min_date"] >= dt.date(2011, 12, 1)
     and stats["max_date"] <= dt.date.today(),
     f"{stats['min_date']} to {stats['max_date']}", "2011-12-01 to today"),
    ("every product maps to a label", "error", not unmapped,
     f"unmapped: {unmapped}" if unmapped else "0 unmapped", "0 unmapped"),
    ("masked text contains no PII patterns", "error", leftover_pii == 0,
     f"{leftover_pii} rows", "0"),
    ("share of complaints with a narrative", "warn", True,
     f"{stats['narrative_share']:.1%}", "reported"),
]
for k in text.PII_PATTERNS:
    rate = stats[f"pii_{k}"] or 0.0
    checks.append((f"unredacted {k} in raw narratives", "warn", rate < 0.01,
                   f"{rate:.3%} of narratives (now masked)", "< 1%"))

run_at = dt.datetime.now()
dq = spark.createDataFrame(
    [(run_at, name, sev, ok, obs, thr) for name, sev, ok, obs, thr in checks],
    "run_at timestamp, check_name string, severity string, passed boolean, "
    "observed string, threshold string",
)
dq.write.mode("append").saveAsTable(T("dq_results"))
display(dq)

failed = [c[0] for c in checks if c[1] == "error" and not c[2]]
if failed:
    raise AssertionError(f"Data-quality errors: {failed}. See dq_results. "
                         "For unmapped products, add a rule in complaint_kit/labels.py.")
print(f"All error-level checks passed. silver rows: {n_silver:,}")
