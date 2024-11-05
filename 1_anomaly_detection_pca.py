# Databricks notebook source
# MAGIC %md
# MAGIC # Anomaly detection on embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we will explore Principal Component Analysis (PCA) as a method for identifying outliers in product descriptions within vendors by analyzing the embeddings of these descriptions. 
# MAGIC <br>PCA offers a robust technique for dimensionality reduction, which can potentially reveal anomalies based on reconstruction errors in the embedding space. 
# MAGIC <br>It is crucial to recognize that PCA is just one of many available algorithms for outlier detection, each with its own set of advantages and limitations. Alternatives like t-SNE, UMAP, or nearest neighbors may provide different insights depending on the data and the specific characteristics of the embeddings. Thus, while PCA can be effective, it is not a silver bullet, and a comprehensive evaluation of multiple techniques is essential for robust anomaly detection.

# COMMAND ----------

# MAGIC %md
# MAGIC Compute: We recommend running this notebook on a Databricks single user cluster, runtime version >= 15.3 ML

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data ingestion

# COMMAND ----------

# MAGIC %run ./_resources/0_setup

# COMMAND ----------

df_spark = spark.read.csv(config['vol_data_landing'],
  header=True,
  inferSchema=True,
  sep=",")

# COMMAND ----------

df_spark.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preparation

# COMMAND ----------

import pandas as pd

df = df_spark.toPandas()

# COMMAND ----------

df.info()

# COMMAND ----------

df['embedding_input'] = df['Products']

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

# Function to get embeddings of the given text 
def get_embedding(text):
    response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": [text]})
    embedding = [e['embedding'] for e in response.data]
    return embedding[0] if embedding else []

# COMMAND ----------

# Create embeddings of concatenated strings
df['embeddings'] = df['embedding_input'].apply(get_embedding)

# COMMAND ----------

import numpy as np
embeddings = np.array(df['embeddings'].tolist())
print("Shape of embeddings:", embeddings.shape)

# COMMAND ----------

df.head()

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

embedding_column = "embeddings"

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

from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType

# Create a Spark session
spark = SparkSession.builder.appName("ParallelAnomalyDetection").getOrCreate()

# Convert the detect_anomalies function to a Pandas UDF
@pandas_udf("Vendor string, value double, embedding array<double>, is_anomaly boolean, reconstruction_error double, pca1 double, pca2 double", PandasUDFType.GROUPED_MAP)
def process_batch(group_df: pd.DataFrame) -> pd.DataFrame:
    embedding_column = "embeddings"
    return detect_anomalies(group_df, embedding_column)

# COMMAND ----------

from joblib import Parallel, delayed
import pandas as pd

def process_batch(filtered_df: pd.DataFrame) -> pd.DataFrame:
    # Apply the anomaly detection function in parallel for each group
    results = Parallel(n_jobs=-1)(delayed(detect_anomalies)(group) 
                                  for _, group in filtered_df.groupby('Vendor'))
    
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
    .saveAsTable(f"{config['catalog']}.{config['schema']}.pca_anomaly_detection_synthetic_data")
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results and visualisation

# COMMAND ----------

# Count unique vendors with identified outliers
number_vendors_with_outliers= df_with_anomalies_all[df_with_anomalies_all['is_anomaly']==True]['Vendor'].nunique()
print(f"Total vendors with outliers: {number_vendors_with_outliers}")

# COMMAND ----------

# Function to print products by vendor and if they are marked as anomalous or not
def print_anomalies(df, vendor):
  total_count_outliers = df[(df['Vendor']== vendor) & (df['is_anomaly']== True)]['Products'].count()
  outliers = df[(df['Vendor']== vendor) & (df['is_anomaly']== True)]['Products'].tolist()
  print(f"Total rows with outliers for {vendor}: {total_count_outliers}")
  print(f"Product outliers for {vendor}: {set(outliers)}")

  anomalies_vendor =  df[df['Vendor']== vendor][['Vendor','Products','is_anomaly']].drop_duplicates()
  return anomalies_vendor

# COMMAND ----------

import matplotlib.pyplot as plt

def plot_pca_scatter(df_input, vendor):
    plt.figure(figsize=(8, 6))
    
    df = df_input[df_input['Vendor']== vendor]

    # Scatter plot of PCA components with reconstruction error
    scatter = plt.scatter(df['pca1'], df['pca2'], c=df['reconstruction_error'], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Reconstruction Error')
    plt.title('PCA of Product Embeddings with Reconstruction Error')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    # Annotate the plot with product names
    for i, row in df.iterrows():
        plt.annotate(row['Products'], (row['pca1'], row['pca2']),
                     fontsize=9, alpha=0.6, color='black')

    plt.show()

# COMMAND ----------

plot_pca_scatter(df_with_anomalies_all,'Vendor 7')

# COMMAND ----------

print_anomalies(df_with_anomalies_all,'Vendor 7')

# COMMAND ----------

plot_pca_scatter(df_with_anomalies_all,'Vendor 1')

# COMMAND ----------

print_anomalies(df_with_anomalies_all,'Vendor 1')