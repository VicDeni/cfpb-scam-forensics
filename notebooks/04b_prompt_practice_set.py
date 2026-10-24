# Databricks notebook source
# MAGIC %md
# MAGIC # 04b · Practice set for improving prompts
# MAGIC Exports two files from the **training months** to the volume's `exports` folder:
# MAGIC * `dev_sample.csv`: up to 100 complaints per product for trying new prompts. Tune prompts
# MAGIC   here, then score the final version once on `eval_sample.csv`. Tuning on the same
# MAGIC   complaints you report on makes a prompt look better than it really is.
# MAGIC * `example_candidates.csv`: 5 short complaints per product to hand-pick few-shot examples
# MAGIC   from. None of them are in the practice set or the evaluation set.
# MAGIC
# MAGIC Run after `04_baseline_model`. It doesn't retrain anything, so it's quick.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(".."))

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "cfpb")
dbutils.widgets.text("dev_per_class", "100")
dbutils.widgets.text("candidates_per_class", "5")
dbutils.widgets.text("llm_max_chars", "4000")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/raw_files"
DEV_PER_CLASS = int(dbutils.widgets.get("dev_per_class"))
CANDIDATES_PER_CLASS = int(dbutils.widgets.get("candidates_per_class"))
LLM_MAX_CHARS = int(dbutils.widgets.get("llm_max_chars"))
SEED = 7  # different from notebook 04's seed, so these are fresh draws


def T(name):
    return f"{CATALOG}.{SCHEMA}.{name}"

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

from complaint_kit.text import truncate

train = spark.table(T("gold_model_dataset")).filter("split = 'train'")

w = Window.partitionBy("product_label").orderBy(F.rand(SEED))
dev = (train.withColumn("_rn", F.row_number().over(w))
       .filter(F.col("_rn") <= DEV_PER_CLASS).drop("_rn").toPandas()
       .sample(frac=1, random_state=SEED).reset_index(drop=True))

# Candidates: short, readable complaints that are not in the practice set.
short = (train.filter(F.length("narrative_clean").between(200, 600))
         .join(spark.createDataFrame(dev[["complaint_id"]]), "complaint_id", "left_anti"))
w2 = Window.partitionBy("product_label").orderBy(F.rand(SEED + 1))
candidates = (short.withColumn("_rn", F.row_number().over(w2))
              .filter(F.col("_rn") <= CANDIDATES_PER_CLASS).drop("_rn")
              .orderBy("product_label").toPandas())

print(f"practice set: {len(dev):,} complaints | candidates: {len(candidates)}")
print(dev["product_label"].value_counts().to_string())

# COMMAND ----------

export_dir = f"{VOLUME_ROOT}/exports"
os.makedirs(export_dir, exist_ok=True)

(dev.assign(text=dev["narrative_clean"].map(lambda t: truncate(t, LLM_MAX_CHARS)))
 [["complaint_id", "product_label", "text"]]
 .to_csv(f"{export_dir}/dev_sample.csv", index=False))

(candidates.rename(columns={"product_label": "label", "narrative_clean": "text"})
 [["complaint_id", "label", "text"]]
 .to_csv(f"{export_dir}/example_candidates.csv", index=False))

print(f"Wrote dev_sample.csv and example_candidates.csv to {export_dir}")
