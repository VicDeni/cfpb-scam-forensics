# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Creates the schema and the volume where the raw CFPB export files go.
# MAGIC
# MAGIC **After running this notebook:** open **Catalog → workspace → cfpb → raw_files**, open the
# MAGIC `zips` folder, click **Upload to this volume**, and upload the CCDB export `.zip` files
# MAGIC you downloaded from the CFPB archive page. Then run `01_ingest_bronze`.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(".."))  # repo root, so `complaint_kit` can be imported

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "cfpb")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/raw_files"

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA} "
          "COMMENT 'CFPB consumer complaints: payment scam forensics and routing project'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw_files "
          "COMMENT 'Raw CFPB export files (zips), extracted CSVs, and exports for the LLM step'")

for folder in ["zips", "extracted", "exports"]:
    os.makedirs(f"{VOLUME_ROOT}/{folder}", exist_ok=True)

print(f"Upload the export zips to: {VOLUME_ROOT}/zips")
print("Already there:", sorted(os.listdir(f"{VOLUME_ROOT}/zips")))
