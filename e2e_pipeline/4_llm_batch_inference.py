# Databricks notebook source
# Define catalog, schema and table names
catalog_name = 'kyra_wulffert'
schema_name = 'anomaly_detection'
input_table_name = 'pca_anomaly_detection'
output_table_name = 'hybrid_anomaly_detection'
model_name = 'model_llmmetamodel'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the data with vendors with anomalies detected by the PCA model

# COMMAND ----------

table_path = f"{catalog_name}.{schema_name}.{input_table_name}"
spark_df = spark.table(table_path)


# COMMAND ----------

df = spark_df.toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

# List of vendors with purchases identified as anomalous by the PCA model
vendors_with_anomalies = df[df['is_anomaly'] == True]['VENDOR_NAME'].unique()

# Create a dictionary to hold the LLM input for each vendor with anomalies
llm_inputs = {}

# Loop through each vendor with anomalies and get a list of unique products
for vendor in vendors_with_anomalies:
    unique_products = df[df['VENDOR_NAME'] == vendor]['PSG_Description'].unique().tolist()
       
    # Add to dictionary
    llm_inputs[vendor] = unique_products

# Display the LLM inputs
for unique_products in llm_inputs.items():
    print(unique_products)
    print("-" * 50) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the registered model for batch inference

# COMMAND ----------

import mlflow
import mlflow.pyfunc

model_name = f"{catalog_name}.{schema_name}.{model_name}"
model_version = "1"
model_uri = f"models:/{model_name}/{model_version}" 
loaded_model = mlflow.pyfunc.load_model(model_uri)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Pandas UDF to run batch inference efficiently

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, MapType, ArrayType


schema = StructType([
    StructField("vendor", StringType(), True),
    StructField("model_prediction", IntegerType(), True),
    StructField("response", StringType(), True),
])

@pandas_udf(schema)
def predict_udf(input_batch: pd.DataFrame) -> pd.DataFrame:
    import mlflow

    # Load the model inside the UDF to avoid serialization issues
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Unpack the struct fields
    input_data = pd.DataFrame({
        "vendor": input_batch["vendor"],
        "products": input_batch["products"].apply(lambda x: eval(x) if isinstance(x, str) else x),
    })

    # Apply the MLflow model
    predictions = loaded_model.predict(input_data)
    return pd.DataFrame(predictions)  # Convert to Pandas DataFrame



# COMMAND ----------

# MAGIC %md
# MAGIC ## Run batch inference
# MAGIC For testing purposes, we set a limit of 100 rows. Change the limit for batch inference on the whole dataset.

# COMMAND ----------

from pyspark.sql.functions import struct, col
import pandas as pd

# Take the first 10 samples
test_llm_inputs = dict(list(llm_inputs.items())[:100])

# Convert to DataFrame
test_data = pd.DataFrame([
    {"vendor": vendor, "products": products}
    for vendor, products in test_llm_inputs.items()
])

spark_test_data = spark.createDataFrame(test_data)

# Use struct to combine columns into a single input for the Pandas UDF
predictions_df = spark_test_data.withColumn(
    "prediction_output",
    predict_udf(struct("vendor", "products"))
)

# Show results
display(predictions_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save results

# COMMAND ----------

from pyspark.sql.functions import col, from_json, explode
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType

# Define the schema of the response JSON
response_schema = StructType([
    StructField("outliers", ArrayType(StringType()), True),
    StructField("is_outlier", IntegerType(), True),
    StructField("reason", StringType(), True),
])

# Extract and parse the `response` field from `prediction_output`
parsed_df = predictions_df.select(
    col("vendor"),
    col("products"),
    col("prediction_output.vendor").alias("output_vendor"),
    # col("prediction_output.model_prediction").alias("model_prediction"),
    from_json(col("prediction_output.response"), response_schema).alias("parsed_response")
)

# Explode the `outliers` array and select fields for the final table
final_df = parsed_df.select(
    col("vendor"),
    col("products"),
    col("output_vendor"),
    # col("model_prediction"),
    col("parsed_response.is_outlier").alias("is_outlier"),
    col("parsed_response.reason").alias("reason"),
    col("parsed_response.outliers").alias("outlier_item")
)

# Show the flattened results
display(final_df)

# COMMAND ----------

# Saving output in our catalog

_=(
    final_df
    .write
    .format("delta")
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable(f"{catalog_name}.{schema_name}.{output_table_name}")
  )
