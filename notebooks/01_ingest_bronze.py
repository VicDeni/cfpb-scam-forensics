# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest → bronze
# MAGIC Unzips the uploaded CFPB exports and loads every record, unchanged, into `bronze_complaints`.
# MAGIC Column names are normalized to snake_case; values stay as raw strings. Each row keeps the
# MAGIC file it came from (`source_file`) so any number can be traced back to its source.

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

# MAGIC %md ## Unzip (skips files already extracted)

# COMMAND ----------

import shutil
import zipfile

zip_dir, out_dir = f"{VOLUME_ROOT}/zips", f"{VOLUME_ROOT}/extracted"
zips = sorted(f for f in os.listdir(zip_dir) if f.lower().endswith(".zip"))
assert zips, f"No zip files found in {zip_dir}. Upload the CFPB exports first (see 00_setup)."

for name in zips:
    target = f"{out_dir}/{name[:-4]}"
    if os.path.exists(f"{target}/_DONE"):
        continue
    os.makedirs(target, exist_ok=True)
    with zipfile.ZipFile(f"{zip_dir}/{name}") as zf:
        for member in zf.infolist():
            if member.is_dir() or member.filename.startswith("__MACOSX"):
                continue
            dest = f"{target}/{os.path.basename(member.filename)}"
            with zf.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    open(f"{target}/_DONE", "w").close()
    print("extracted", name)

data_files = sorted(
    f"{root}/{f}"
    for root, _, files in os.walk(out_dir)
    for f in files
    if f.lower().endswith((".csv", ".json"))
)
print(f"{len(data_files)} data files:")
for f in data_files:
    print(" ", f)

# COMMAND ----------

# MAGIC %md ## Read each file and union them

# COMMAND ----------

from functools import reduce

from pyspark.sql import functions as F

from complaint_kit.schema import missing_required, normalize_column_name


def read_export(path):
    if path.lower().endswith(".csv"):
        # Narratives contain commas, quotes and line breaks.
        df = (spark.read.option("header", True).option("multiLine", True)
              .option("escape", '"').csv(path))
    else:
        df = spark.read.option("multiLine", True).json(path)
    df = df.toDF(*[normalize_column_name(c) for c in df.columns])
    return df.select(*[F.col(c).cast("string") for c in df.columns]).withColumn(
        "source_file", F.lit(path)
    )


frames = [read_export(p) for p in data_files]
for path, df in zip(data_files, frames):
    missing = missing_required(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns {missing}.\nColumns found: {df.columns}\n"
            "If a column just has a different name, add it to COLUMN_ALIASES in complaint_kit/schema.py."
        )

bronze = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), frames)
bronze = bronze.withColumn("ingested_at", F.current_timestamp())

# COMMAND ----------

(bronze.write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(T("bronze_complaints")))

counts = spark.table(T("bronze_complaints")).groupBy("source_file").count().orderBy("source_file")
display(counts)
print("bronze rows:", spark.table(T("bronze_complaints")).count())
