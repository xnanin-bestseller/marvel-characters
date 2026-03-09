# Databricks notebook source
# MAGIC %pip install marvel_characters-1.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import time
import os
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from mlflow import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError
from dotenv import load_dotenv

from marvel_characters.config import ProjectConfig
from marvel_characters.serving.model_serving import ModelServing
from marvel_characters.utils import is_databricks


# COMMAND ----------
# spark session
spark = SparkSession.builder.getOrCreate()

w = WorkspaceClient()

print(f"Authenticated Databricks user: {w.current_user.me().user_name}")

os.environ["DBR_HOST"] = w.config.host

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
# Initialize model serving
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.marvel_character_model_custom", endpoint_name="marvel-character-model-serving"
)

# COMMAND ----------
# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()


# COMMAND ----------
# Create a sample request body
required_columns = [
    "Height",
    "Weight",
    "Universe",
    "Identity",
    "Gender",
    "Marital_Status",
    "Teams",
    "Origin",
    "Magic",
    "Mutant"]


# Sample 1000 records from the training set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# Sample records from the training set
sampled_records = test_set[required_columns].sample(n=18000, replace=True)

# Replace NaN values with None (which will be serialized as null in JSON)
import numpy as np
sampled_records = sampled_records.replace({np.nan: None}).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------
# MAGIC %md
# MAGIC ### Connection approach update
# MAGIC
# MAGIC This tutorial now calls the serving endpoint through the Databricks SDK:
# MAGIC `w.serving_endpoints.query(...)`.
# MAGIC
# MAGIC Instead of manually posting HTTP requests with an `Authorization: Bearer ...` header,
# MAGIC the SDK uses Databricks authentication resolution (notebook context, profile, and env vars),
# MAGIC which is generally more robust and easier to troubleshoot.

# COMMAND ----------
# Call the endpoint with one sample record

"""
Each dataframe record in the request body should be list of json with columns looking like:

[{'Height': 1.75,
  'Weight': 70.0,
  'Universe': 'Earth-616',
  'Identity': 'Public',
  'Gender': 'Male',
  'Marital_Status': 'Single',
  'Teams': 0, # Teams is an Integer
  'Origin': 'Human',
  'Magic': 1,
  'Mutant': 1}]
"""

def call_endpoint(record):
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = f"{os.environ['DBR_HOST']}/serving-endpoints/marvel-character-model-serving/invocations"
    print(f"Calling endpoint: {serving_endpoint}")

    # Connection approach changed: we now use the Databricks SDK query API
    # instead of a manual HTTP request with Authorization headers.
    # This relies on the SDK auth chain (profile/env/notebook context),
    # which is typically more reliable than managing bearer tokens manually.
    try:
        response = w.serving_endpoints.query(
            name="marvel-character-model-serving",
            dataframe_records=record,
        )
        return 200, response
    except DatabricksError as e:
        status_code = getattr(e, "status_code", None)
        error_code = getattr(e, "error_code", None)
        message = str(e)

        if status_code is None:
            if "Invalid Token" in message or "UNAUTHENTICATED" in message:
                status_code = 401
            elif "PERMISSION_DENIED" in message:
                status_code = 403
            elif "not found" in message.lower():
                status_code = 404
            else:
                status_code = 500

        return status_code, {"error_code": error_code, "message": message}
    except Exception as e:
        return 500, {"error_code": "UNKNOWN", "message": str(e)}


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# Load test
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2) 
# COMMAND ----------


