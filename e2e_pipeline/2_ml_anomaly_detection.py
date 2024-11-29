# Databricks notebook source
# MAGIC %md
# MAGIC ## Data ingestion

# COMMAND ----------

import time

start_time = time.time()

# COMMAND ----------

# Define catalog, schema and table names
catalog_name = 'kyra_wulffert'
schema_name = 'anomaly_detection'
table_name = 'purchase_orders_processed_test'
groupby_field = 'VENDOR_NAME'
product_description_field = 'PSG_Description'


# COMMAND ----------

df_spark = spark.read.format('delta').table(f'{catalog_name}.{schema_name}.{table_name}')

# COMMAND ----------

df_spark.show(5)

# COMMAND ----------

import pandas as pd

df = df_spark.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply Principal Component Analysis on Embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC We define a function to detect anomalies using PCA. As threshold we mark as outliers the points with reconstruction error above 99.9%. Fine tuning on the threshold is needed based on the use case needs.

# COMMAND ----------

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

embedding_column = "embedding_output"

def detect_anomalies(group_df: pd.DataFrame) -> pd.DataFrame:
  embeddings = np.vstack(group_df[embedding_column])
  
  # Check if the number of samples is greater than 1
  if embeddings.shape[0] > 1 and embeddings.shape[1] > 1:
      # Scale embeddings       
      scaler = StandardScaler()
      embeddings_scaled = scaler.fit_transform(embeddings)
      # Apply PCA
      pca = PCA(n_components=min(embeddings_scaled.shape[0], embeddings_scaled.shape[1], 2), svd_solver='randomized', random_state=23)
      reduced_embeddings = pca.fit_transform(embeddings_scaled)
      
      # Reconstruct the embeddings
      reconstructed_embeddings = pca.inverse_transform(reduced_embeddings)
      
      # Calculate the reconstruction error
      reconstruction_error = np.mean((embeddings_scaled - reconstructed_embeddings) ** 2, axis=1)
      
      # Define a threshold for anomalies (e.g., top 1% as anomalies)
      threshold = np.percentile(reconstruction_error, 99)
      
      # Identify anomalies
      group_df['is_anomaly'] = reconstruction_error > threshold
      group_df['reconstruction_error'] = reconstruction_error
      group_df[['pca1', 'pca2']] = reduced_embeddings
  else:
      # If not enough samples, mark all as non-anomalous or handle differently
      group_df['is_anomaly'] = False
      group_df['reconstruction_error'] = np.nan
      group_df[['pca1', 'pca2']] = [np.nan, np.nan]
  
  return group_df

# COMMAND ----------

# MAGIC %md
# MAGIC Function to run the anomaly detector on all our data in an efficient way.

# COMMAND ----------

from joblib import Parallel, delayed
import pandas as pd

def process_batch(filtered_df: pd.DataFrame) -> pd.DataFrame:
    # Apply the anomaly detection function in parallel for each group
    results = Parallel(n_jobs=-1)(delayed(detect_anomalies)(group) 
                                  for _, group in filtered_df.groupby(groupby_field))
    
    # Concatenate results
    return pd.concat(results).reset_index(drop=True)

# COMMAND ----------

df_with_anomalies_all = process_batch(df)

# COMMAND ----------

df_with_anomalies_all.shape

# COMMAND ----------

# Saving PCA output in our catalog
_=(
spark.createDataFrame(df_with_anomalies_all)
    .write
    .format("delta")
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable(f"{catalog_name}.{schema_name}.pca_anomaly_detection_output")
  )

# COMMAND ----------

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results and visualisation

# COMMAND ----------

# Count unique vendors with identified outliers
number_vendors_with_outliers= df_with_anomalies_all[df_with_anomalies_all['is_anomaly']==True][groupby_field].nunique()
print(f"Total vendors with outliers: {number_vendors_with_outliers}")
print(f"Total number of vendors: {df[groupby_field].nunique()}")

# COMMAND ----------

df_with_anomalies_all[df_with_anomalies_all['is_anomaly']==True][groupby_field].unique()

# COMMAND ----------

# Function to print products by vendor and if they are marked as anomalous or not
def print_anomalies(df, vendor):
  total_count_outliers = df[(df[groupby_field]== vendor) & (df['is_anomaly']== True)][product_description_field].count()
  outliers = df[(df[groupby_field]== vendor) & (df['is_anomaly']== True)][product_description_field].tolist()
  print(f"Total rows with outliers for {vendor}: {total_count_outliers}")
  print(f"Product outliers for {vendor}: {set(outliers)}")

  anomalies_vendor =  df[df[groupby_field]== vendor][[groupby_field,product_description_field,'is_anomaly']].drop_duplicates()
  return anomalies_vendor

# COMMAND ----------

print_anomalies(df_with_anomalies_all,'Supplier 10010')

# COMMAND ----------

print_anomalies(df_with_anomalies_all,'Supplier 5031')
