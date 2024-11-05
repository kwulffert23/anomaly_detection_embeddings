# Databricks notebook source
# MAGIC %md
# MAGIC # Anomaly Detection Hybrid Solution

# COMMAND ----------

# MAGIC %md
# MAGIC ![pipeline_emb.png](./data/pipeline_emb.png "pipeline_emb.png")

# COMMAND ----------

# MAGIC %md
# MAGIC Although the PCA model detects anomalous purchases, there might be identified purchases that are not anomalous giving a context or a set of instructions. The solution can be further improved by instructing an LLM to evaluate the vendor with anomalous transactions based on the PCA model score and to provide a reason for the evaluation output on whether products purchased from those vendors are identified as anomalous or not.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up the environment and functions

# COMMAND ----------

# MAGIC %md
# MAGIC Compute: We recommend running this notebook on a Databricks single user cluster, runtime version >= 15.3 ML

# COMMAND ----------

# DBTITLE 1,Install libraries used in this demo
# MAGIC %pip install --upgrade openai tenacity tqdm
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/0_setup

# COMMAND ----------

# DBTITLE 1,Select model endpoint
# The endpoint ID of the model to use. Not all endpoints support function calling.
MODEL_ENDPOINT_ID = "databricks-meta-llama-3-1-70b-instruct"

# COMMAND ----------

# DBTITLE 1,Set up API client
import concurrent.futures
import pandas as pd
from openai import OpenAI, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
)  # for exponential backoff
from tqdm.notebook import tqdm
from typing import List, Optional


# A token and the workspace's base FMAPI URL are needed to talk to endpoints
fmapi_token = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .getOrElse(None)
)
fmapi_base_url = (
    f'https://{spark.conf.get("spark.databricks.workspaceUrl")}/serving-endpoints'
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The following defines helper functions that assist the LLM to respond according to the specified schema.

# COMMAND ----------

# DBTITLE 1,Set up helper functions

openai_client = OpenAI(api_key=fmapi_token, base_url=fmapi_base_url)


# NOTE: We *strongly* recommend handling retry errors with backoffs, so your code gracefully degrades when it bumps up against pay-per-token rate limits.
@retry(
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception(RateLimitError),
)

def call_chat_model(
    prompt: str, temperature: float = 0.0, max_tokens: int = 100, **kwargs
):
    """Calls the chat model and returns the response text or tool calls."""
    chat_args = {
        "model": MODEL_ENDPOINT_ID,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    chat_args.update(kwargs)

    try:
        chat_completion = openai_client.chat.completions.create(**chat_args)
        response = chat_completion.choices[0].message

        if response.tool_calls:
            call_args = [c.function.arguments for c in response.tool_calls]
            if len(call_args) == 1:
                return call_args[0]
            return call_args
        
        return response.content  
    except Exception as e:
        # print(f"Error: {e}")
        return None
    
def call_in_parallel(func, prompts: List[str]) -> List:
    """Calls func(p) for all prompts in parallel and returns responses."""
    # This uses a relatively small thread pool to avoid triggering default workspace rate limits.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = []
        for r in tqdm(executor.map(func, prompts), total=len(prompts)):
            results.append(r)
        return results


def results_to_dataframe(products: List[str], responses: List[str]):
    """Combines reviews and model responses into a dataframe for tabular display."""
    return pd.DataFrame({"Products list": products, "Model response": responses})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the data with vendors with anomalies detected by the PCA model

# COMMAND ----------

df_spark = spark.table(f"{config['catalog']}.{config['schema']}.pca_anomaly_detection_synthetic_data")
df = df_spark.toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

# List of vendors with purchases identified as anomalous by the PCA model
vendors_with_anomalies = df[df['is_anomaly'] == True]['Vendor'].unique()

# Create a dictionary to hold the LLM input for each vendor with anomalies
llm_inputs = {}

# Loop through each vendor with anomalies and get a list of unique products
for vendor in vendors_with_anomalies:
    unique_products = df[df['Vendor'] == vendor]['Products'].unique().tolist()
       
    # Add to dictionary
    llm_inputs[vendor] = unique_products

# Display the LLM inputs
for unique_products in llm_inputs.items():
    print(unique_products)
    print("-" * 50) 

# COMMAND ----------

# MAGIC %md
# MAGIC # Anomaly detection with LLMs and tools

# COMMAND ----------

#  Define tools to have a fixed output format for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "_outlier_detection",
            "description": "Identifies outliers in the list of products",
            "parameters": {
                "type": "object",
                "properties": {
                    "outliers": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for the item to be identified as an anomaly"
                    },                    
                },
                "required": [
                    "outliers",
                    "reason"
                ],
            },
        },
    },
]


def prompt_with_outlier_tool(products: List[str]):
    # Convert the list of products to a string format suitable for the LLM
    products_str = "\n".join(products)
    prompt = OUTLIER_PROMPT_TEMPLATE.format(products=products_str)
    return call_chat_model(prompt, tools=tools)


# COMMAND ----------

#  define a prompt with few examples

OUTLIER_PROMPT_TEMPLATE = """Imagine you are analyzing a list of products offered by a fictional company. The company specializes in certain types of products, but the specifics are not given. Use reasonable assumptions about what types of products a company with a given focus might offer. Your task is to identify any outliers in the list—products that do not fit well with what the company would typically provide. Consider the context and relationships between products, and avoid identifying outliers based on overly narrow or literal interpretations. If a product could reasonably be related to the company's focus or is a natural extension of their offerings, do not consider it an outlier. If all products could reasonably fit into a general theme or focus area, do not list any outliers.
DO NOT consider as outliers any transportation, delivery, courier, freight services, or charges as they are usually related to the delivery of parts a company produces.
Your output should be in JSON format.

Examples:
- Products: ["apple", "kiwi", "strawberry", "bread", "COURIER SERVICES"], Output: {{"outliers": ["bread"], "reason": "bread is not a type of fruit, which is what the company seems to specialize in."}}
- Products: ["chair", "table", "sofa", "bed", "refrigerator", "tree"], Output: {{"outliers": ["refrigerator", "tree"], "reason": "refrigerator and tree are not typical furniture items, which appears to be the company's focus."}}
- Products: ["Delivery Charges", "floor", "wall", "ceiling", "door", "window"], Output: {{"outliers": []}}

# Products
{products}

"""


# COMMAND ----------

llm_inputs.values()

# COMMAND ----------

PRODUCT_INPUTS = list(llm_inputs.values())
# We apply the new prompt with examples to the functions we defined already
pd.set_option('display.max_colwidth', None)
results = call_in_parallel(prompt_with_outlier_tool, PRODUCT_INPUTS)
df_results = results_to_dataframe(PRODUCT_INPUTS, results)

# COMMAND ----------

df_results

# COMMAND ----------

