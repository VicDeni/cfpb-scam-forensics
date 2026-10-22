# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Baseline model + export the LLM evaluation set
# MAGIC * Trains TF-IDF + logistic regression on the train split and logs it to **MLflow**.
# MAGIC * Scores it two ways: on the test months as they really are (credit reporting dominates),
# MAGIC   and on a **balanced evaluation sample** (up to 200 per class) that the LLM will also see.
# MAGIC * Exports `eval_sample.csv` and `few_shot_examples.csv` to the volume's `exports` folder.
# MAGIC   Download both and run `scripts/run_llm_eval.py` on your laptop.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(".."))

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "cfpb")
dbutils.widgets.text("max_train_rows", "200000")
dbutils.widgets.text("max_test_rows", "100000")
dbutils.widgets.text("eval_per_class", "200")
dbutils.widgets.text("llm_max_chars", "4000")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/raw_files"
MAX_TRAIN = int(dbutils.widgets.get("max_train_rows"))
MAX_TEST = int(dbutils.widgets.get("max_test_rows"))
EVAL_PER_CLASS = int(dbutils.widgets.get("eval_per_class"))
LLM_MAX_CHARS = int(dbutils.widgets.get("llm_max_chars"))
SEED = 42


def T(name):
    return f"{CATALOG}.{SCHEMA}.{name}"

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

ds = spark.table(T("gold_model_dataset"))


def random_rows(df, n):
    total = df.count()
    frac = min(1.0, 1.2 * n / total) if total else 1.0
    return df.sample(fraction=frac, seed=SEED).limit(n).toPandas()


train = random_rows(ds.filter("split = 'train'"), MAX_TRAIN)
test = random_rows(ds.filter("split = 'test'"), MAX_TEST)

w = Window.partitionBy("product_label").orderBy(F.rand(SEED))
eval_sample = (ds.filter("split = 'test'").withColumn("_rn", F.row_number().over(w))
               .filter(F.col("_rn") <= EVAL_PER_CLASS).drop("_rn").toPandas()
               .sample(frac=1, random_state=SEED).reset_index(drop=True))  # mixed order for --limit runs

print(f"train {len(train):,} | test {len(test):,} | balanced eval sample {len(eval_sample):,}")
print(eval_sample["product_label"].value_counts().to_string())

# COMMAND ----------

import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from complaint_kit import evaluate

params = {"ngram_range": (1, 2), "min_df": 5, "max_features": 200_000,
          "C": 4.0, "class_weight": "balanced", "max_iter": 1000}

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=params["ngram_range"], min_df=params["min_df"],
                              max_features=params["max_features"], sublinear_tf=True)),
    ("clf", LogisticRegression(C=params["C"], class_weight=params["class_weight"],
                               max_iter=params["max_iter"])),
])

with mlflow.start_run(run_name="baseline_tfidf_logreg") as run:
    pipe.fit(train["narrative_clean"], train["product_label"])
    test_metrics = evaluate.summarize(test["product_label"], pipe.predict(test["narrative_clean"]))
    eval_pred = pipe.predict(eval_sample["narrative_clean"])
    eval_metrics = evaluate.summarize(eval_sample["product_label"], eval_pred)

    mlflow.log_params({k: str(v) for k, v in params.items()} | {
        "train_rows": len(train), "test_rows": len(test), "eval_rows": len(eval_sample)})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if k != "n"})
    mlflow.log_metrics({f"eval_{k}": v for k, v in eval_metrics.items() if k != "n"})
    try:
        mlflow.sklearn.log_model(pipe, "model")
    except Exception as exc:  # metrics are what matter; don't fail on artifact limits
        print("Skipped saving model artifact:", exc)

print("Test months (natural mix):", test_metrics)
print("Balanced eval sample:     ", eval_metrics)
print("MLflow run:", run.info.run_id)

# COMMAND ----------

import pandas as pd

display(pd.DataFrame(evaluate.per_class(eval_sample["product_label"], eval_pred)))
display(pd.DataFrame(evaluate.top_confusions(eval_sample["product_label"], eval_pred)))

preds = eval_sample[["complaint_id", "product_label"]].assign(predicted=eval_pred)
spark.createDataFrame(preds).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(T("baseline_eval_predictions"))

# COMMAND ----------

# MAGIC %md ## Export files for the LLM step

# COMMAND ----------

from complaint_kit.sampling import pick_examples
from complaint_kit.text import truncate

export_dir = f"{VOLUME_ROOT}/exports"
os.makedirs(export_dir, exist_ok=True)

(eval_sample.assign(text=eval_sample["narrative_clean"].map(lambda t: truncate(t, LLM_MAX_CHARS)))
 [["complaint_id", "product_label", "text"]]
 .to_csv(f"{export_dir}/eval_sample.csv", index=False))

# Few-shot examples come from TRAIN only, so no test complaint appears in a prompt.
examples = pick_examples(train, "product_label", "narrative_clean", per_class=1, seed=SEED)
pd.DataFrame(examples).to_csv(f"{export_dir}/few_shot_examples.csv", index=False)

print(f"Wrote {len(eval_sample)} eval rows and {len(examples)} examples to {export_dir}")
print("Download both from Catalog → raw_files → exports, then run scripts/run_llm_eval.py locally.")
