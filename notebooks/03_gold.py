# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold tables + documentation
# MAGIC * `gold_model_dataset`: complaints with a narrative and a usable label, split **by time**.
# MAGIC   The newest months are the test set, so models are scored on complaints from after
# MAGIC   everything they were trained on (no leakage from the future).
# MAGIC * The CFPB published narratives weeks after a complaint arrived, then stopped publishing in
# MAGIC   August 2026, so the last months in the data have few or no narratives. Months with under
# MAGIC   25% of the peak month's narrative count are treated as incomplete and left out of modeling.
# MAGIC * `gold_complaint_trends` and `gold_top_companies` feed the dashboard.
# MAGIC * Writes the data dictionary into Unity Catalog as table and column comments.
# MAGIC   Open any table in **Catalog Explorer** to see them. If your workspace shows a **Lineage**
# MAGIC   tab there, it traces each table back to the ones it was built from.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(".."))

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "cfpb")
dbutils.widgets.text("test_months", "2")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TEST_MONTHS = int(dbutils.widgets.get("test_months"))


def T(name):
    return f"{CATALOG}.{SCHEMA}.{name}"

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

from complaint_kit.governance import comment_statements
from complaint_kit.labels import LABELS, UNMAPPED

silver = spark.table(T("silver_complaints")).withColumn(
    "month", F.trunc("date_received", "month")
)
narratives_by_month = silver.filter("has_narrative").groupBy("month").count().orderBy("month")
display(narratives_by_month)
peak = narratives_by_month.agg(F.max("count")).first()[0]
last_month = (narratives_by_month.filter(F.col("count") >= 0.25 * peak)
              .agg(F.max("month")).first()[0])
test_start = spark.sql(f"SELECT add_months(DATE'{last_month}', {1 - TEST_MONTHS})").first()[0]
print(f"Last month with complete narratives: {last_month}. "
      f"Test set: {test_start} to {last_month}; later months are left out of modeling.")

# COMMAND ----------

model_ds = (silver
            .filter(F.col("has_narrative") & F.col("product_label").isin(list(LABELS)))
            .filter(F.length("narrative_clean") >= 30)
            .filter(F.col("month") <= F.lit(last_month))
            .withColumn("split", F.when(F.col("month") >= F.lit(test_start), "test")
                        .otherwise("train"))
            .select("complaint_id", "date_received", "product_label", "narrative_clean", "split"))
model_ds.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(T("gold_model_dataset"))

display(spark.table(T("gold_model_dataset"))
        .groupBy("product_label").pivot("split", ["train", "test"]).count()
        .orderBy(F.desc("train")))

# COMMAND ----------

trends = (silver.filter(F.col("product_label") != UNMAPPED)
          .groupBy("month", "product_label")
          .agg(F.count("*").alias("n_complaints"),
               F.sum(F.col("has_narrative").cast("int")).alias("n_with_narrative"),
               F.avg(F.col("timely_response").cast("double")).alias("pct_timely")))
trends.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(T("gold_complaint_trends"))

rank = Window.partitionBy("product_label").orderBy(F.desc("n_complaints"))
top_companies = (silver.filter(F.col("product_label").isin(list(LABELS)))
                 .groupBy("product_label", "company").agg(F.count("*").alias("n_complaints"))
                 .withColumn("rank", F.row_number().over(rank)).filter("rank <= 10"))
top_companies.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(T("gold_top_companies"))

# COMMAND ----------

# MAGIC %md ## Data dictionary → Unity Catalog comments

# COMMAND ----------

for key in ["bronze_complaints", "silver_complaints", "gold_model_dataset",
            "gold_complaint_trends", "dq_results"]:
    for stmt in comment_statements(T(key), key, spark.table(T(key)).columns):
        spark.sql(stmt)
print("Comments written. Open the tables in Catalog Explorer to see them.")