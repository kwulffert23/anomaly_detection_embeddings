# Databricks notebook source
import time

start_time = time.time()

# COMMAND ----------

# Define catalog, schema and table names
catalog_name = 'kyra_wulffert'
schema_name = 'anomaly_detection'
input_table_name = 'purchase_orders'
output_table_name = 'purchase_orders_processed_test'


# COMMAND ----------

# Allows us to reference these values directly in the SQL/Python function creation
dbutils.widgets.text("catalog_name", defaultValue=catalog_name, label="Catalog Name")
dbutils.widgets.text("schema_name", defaultValue=schema_name, label="Schema Name")
dbutils.widgets.text("input_table_name", defaultValue=input_table_name, label="Input Table Name")
dbutils.widgets.text("output_table_name", defaultValue=output_table_name, label="Output Table Name")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a column with the text to embed 
# MAGIC CREATE TABLE IF NOT EXISTS ${catalog_name}.${schema_name}.${output_table_name} AS
# MAGIC SELECT 
# MAGIC     *,
# MAGIC     CONCAT(VENDOR_NAME, '_', PSG_Description) AS embedding_input_string
# MAGIC FROM ${catalog_name}.${schema_name}.${input_table_name};

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Add the embedding column to the table
# MAGIC ALTER TABLE ${catalog_name}.${schema_name}.${output_table_name}
# MAGIC ADD COLUMNS (embedding_output ARRAY<DOUBLE>);
# MAGIC
# MAGIC -- Update the table with embedding inputs and feed them to an embedder to generate the embeddings. TODO: add the name of the provisioned throughput end point you created to replace 'gte_large_pt_kw'.
# MAGIC UPDATE ${catalog_name}.${schema_name}.${output_table_name}
# MAGIC SET embedding_output =
# MAGIC     ai_query(
# MAGIC         'gte_large_pt_kw', 
# MAGIC         embedding_input_string,
# MAGIC         returnType => 'ARRAY<DOUBLE>'
# MAGIC     );

# COMMAND ----------

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
