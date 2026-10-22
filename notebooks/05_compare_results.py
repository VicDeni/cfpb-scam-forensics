# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Compare baseline vs. LLM
# MAGIC Upload `results/llm_results.jsonl` (from `scripts/run_llm_eval.py`) to the volume's
# MAGIC `exports` folder first. This notebook scores every approach **on the same complaints**,
# MAGIC then writes the tables the dashboard uses:
# MAGIC `gold_model_comparison`, `gold_per_class_metrics`, `gold_confusions`, `gold_error_examples`.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(".."))

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "cfpb")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/raw_files"


def T(name):
    return f"{CATALOG}.{SCHEMA}.{name}"

# COMMAND ----------

import pandas as pd

from complaint_kit import evaluate
from complaint_kit.governance import comment_statements

llm_df = pd.read_json(f"{VOLUME_ROOT}/exports/llm_results.jsonl", lines=True,
                      dtype={"complaint_id": str})
base_df = spark.table(T("baseline_eval_predictions")).toPandas()
base_df["complaint_id"] = base_df["complaint_id"].astype(str)

groups = {("baseline_tfidf_logreg", None, None): base_df}
for (variant, version, model), g in llm_df.groupby(["variant", "prompt_version", "model"]):
    groups[(variant, version, model)] = g.rename(columns={"true_label": "product_label"})

# Score everything on the complaints every approach completed.
common = set.intersection(*[set(g["complaint_id"]) for g in groups.values()])
print(f"Comparing {len(groups)} approaches on {len(common):,} shared complaints.")

summary_rows, class_rows, confusion_rows = [], [], []
for (approach, version, model), g in groups.items():
    g = g[g["complaint_id"].isin(common)]
    has_cost = approach != "baseline_tfidf_logreg"
    s = evaluate.summarize(
        g["product_label"].tolist(), g["predicted"].tolist(),
        g["input_tokens"].tolist() if has_cost else None,
        g["output_tokens"].tolist() if has_cost else None,
        g["cost_usd"].fillna(0).tolist() if has_cost else None,
    )
    summary_rows.append({"approach": approach, "prompt_version": version, "model": model, **s})
    for r in evaluate.per_class(g["product_label"], g["predicted"]):
        class_rows.append({"approach": approach, "prompt_version": version, **r})
    for r in evaluate.top_confusions(g["product_label"], g["predicted"], k=15):
        confusion_rows.append({"approach": approach, "prompt_version": version, **r})

comparison = pd.DataFrame(summary_rows)
for col in ["avg_input_tokens", "avg_output_tokens", "cost_per_1k_usd"]:
    if col not in comparison:
        comparison[col] = None
display(comparison)

# COMMAND ----------

def save(pdf, name):
    spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true") \
        .saveAsTable(T(name))


save(comparison, "gold_model_comparison")
save(pd.DataFrame(class_rows), "gold_per_class_metrics")
save(pd.DataFrame(confusion_rows), "gold_confusions")

for stmt in comment_statements(T("gold_model_comparison"), "gold_model_comparison",
                               comparison.columns):
    spark.sql(stmt)

# COMMAND ----------

# MAGIC %md ## Error examples for analysis
# MAGIC Read these. The best interview stories come from *why* the model was wrong
# MAGIC (for example, a debt collector chasing a credit card balance).

# COMMAND ----------

text_df = (spark.table(T("gold_model_dataset")).select("complaint_id", "narrative_clean")
           .toPandas().astype({"complaint_id": str}))
errors = (llm_df[llm_df["true_label"] != llm_df["predicted"]]
          .merge(text_df, on="complaint_id", how="left")
          .assign(snippet=lambda d: d["narrative_clean"].str.slice(0, 400))
          [["variant", "prompt_version", "complaint_id", "true_label", "predicted", "snippet"]])
save(errors, "gold_error_examples")
display(errors.head(50))
