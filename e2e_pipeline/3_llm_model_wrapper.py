# Databricks notebook source
# Define catalog and schema 
catalog_name = 'kyra_wulffert'
schema_name = 'anomaly_detection'

# COMMAND ----------

# MAGIC %md
# MAGIC # Anomaly detection with LLMs and tools
# MAGIC We will define our LLM and tools model as follows:
# MAGIC 1. Define a prompt to instruct the LLM.
# MAGIC 2. Define the response format to enforce a preferred JSON output.
# MAGIC 3. Define an mlflow pyfunc wrapper with all our components (prompt, llm to use, response_format). 
# MAGIC 4. Run an mlflow experiment and log the resulting model.
# MAGIC 5. Load the model to test and proceed with the evaluation step.

# COMMAND ----------

#  define a prompt with few shot examples

OUTLIER_PROMPT_TEMPLATE = """
You are analyzing a list of products procured by the procurement department of a **electricity utility company**. This company typically focuses on items related to its core operations, such as power generation, transmission, and distribution. These include but are not limited to:

- Electrical components (e.g., transformers, circuit breakers, wires, switches)
- Maintenance and repair supplies (e.g., tools, safety gear, lubricants)
- Office supplies and services
- Transportation and delivery charges for parts and equipment

**Your task**: Identify any outliers—products that clearly do not fit well with what an electricity utility company would typically procure. Keep in mind:
1. **Anomalies are extremely rare**: Do not flag items as outliers unless you are highly certain they are unrelated to the electricity utility sector.
2. **Domain relevance**: Use knowledge of the electricity utility industry to make well-informed judgments about what products are typical.
3. **Transportation and delivery services**: Do NOT flag any charges related to transportation, delivery, courier, or freight, as these are often necessary for procurement.
4. **Avoid false positives**: If a product could reasonably fit within the procurement patterns of an electricity utility company, do not flag it as an outlier. Be conservative in identifying anomalies.
5. **Nuance over categories**: Focus on whether a product aligns with the company's purpose rather than its broad category.

**Output format**: Provide your response in JSON format, including:
- A list of flagged outliers (`outliers`).
- A binary flag (`is_outlier`) indicating whether any outliers were identified (`true` if there are outliers, `false` otherwise).
- A detailed explanation (`reason`) for why the outliers were identified or why all items fit.

If there are no anomalies, explicitly state this and set `is_outlier` to `false`.

Examples:
1. Products: ["transformer", "wire", "office chair", "freight services"]
   Output: {{"outliers": [], "is_outlier": false, "reason": "All items fit within typical procurement for an electricity utility company, including office furniture and transportation services."}}

2. Products: ["transformer", "wire", "office chair", "bicycle"]
   Output: {{"outliers": ["bicycle"], "is_outlier": true, "reason": "Bicycles are not typically procured by an electricity utility company."}}

3. Products: ["wrench", "circuit breaker", "safety helmet", "yoga mat"]
   Output: {{"outliers": ["yoga mat"], "is_outlier": true, "reason": "A yoga mat is unrelated to maintenance, electrical components, or typical utility procurement."}}

4. Products: ["electricity tariff analysis software", "data cables", "cleaning services"]
   Output: {{"outliers": [], "is_outlier": false, "reason": "All items could plausibly be used by an electricity utility company for operations, IT, or office maintenance."}}

# Products
{products}

"""

# COMMAND ----------

# Define response format required

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "anomaly_label_reason_extractions",
        "schema": {
            "type": "object",
            "properties": {
                "outliers": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "is_outlier": {
                    "type": "number",
                    "description": "Binary flag indicating whether any outliers were identified (1 if outliers exist, 0 otherwise).",
                    "enum": [0.0, 1.0],
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
        "strict": True,
    },
}

# COMMAND ----------

import mlflow.pyfunc
import openai
import json
from mlflow.utils.databricks_utils import get_databricks_host_creds

class ModelMLflowWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_name, prompt_template, max_tokens=500, response_format="json"):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.response_format = response_format

    def load_context(self, context):
        # Initialize the OpenAI client using Databricks credentials
        creds = get_databricks_host_creds("databricks")
        self.client = openai.OpenAI(
            api_key=creds.token, 
            base_url=f"{creds.host}/serving-endpoints"
        )


    def predict(self, context, model_input):
        # Expect model_input as a DataFrame with 'vendor' and 'products' columns
        results = []
        for _, row in model_input.iterrows():
            vendor = row['vendor']
            products = row['products']
            products_str = "\n".join(products)

            try:
                # Call the LLM API
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user", 
                            "content": self.prompt_template.format(products=products_str)
                        }
                    ],
                    model=self.model_name,
                    response_format=self.response_format,
                    max_tokens=self.max_tokens,
                )
                # Parse response and extract relevant fields
                response_content = json.loads(response.choices[0].message.content)
                is_outlier = response_content.get("is_outlier", 0)  # Default to 0 if missing
                
                results.append({
                    "vendor": vendor,
                    "response": response.choices[0].message.content,
                    "model_prediction": is_outlier  # Add binary prediction at top level
                })
            except Exception as e:
                results.append({
                    "vendor": vendor,
                    "response": f"Error: {str(e)}",
                    "model_prediction": None  
                })
        return results


# COMMAND ----------

from mlflow.models.signature import infer_signature
import pandas as pd

# Example data for signature inference
input_example = pd.DataFrame([
    {"vendor": "VendorA", "products": ["Bricks", "Cement", "Cloud Software"]},
    {"vendor": "VendorB", "products": ["Onion", "Carrot", "Garlic", "Orange"]}
])
# Mock output for signature inference
output_example = [
    {
        "model_prediction": 1.0,  
        "response": '{"outliers": ["Cloud Software"], "is_outlier": 1.0, "reason": "Cloud Software is not related to the construction industry, which is what the company seems to specialize in."}'
    },
    {
        "model_prediction": 1.0,  
        "response": '{"outliers": ["Orange"], "is_outlier": 1.0, "reason": "Orange is a fruit and not a vegetable"}'
    }
]

# Infer signature from the examples
signature = infer_signature(input_example, output_example)

# COMMAND ----------

# Initialize the wrapper
model_name = "databricks-meta-llama-3-1-70b-instruct"
pyfunc_model = ModelMLflowWrapper(
    model_name=model_name,
    prompt_template=OUTLIER_PROMPT_TEMPLATE,
    max_tokens=500,
    response_format=response_format
)

# Log the model to MLflow
import mlflow

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="Model_pyfunc_model",  # Path for the logged model
        python_model=pyfunc_model,  # The wrapper class instance
        registered_model_name="Model_LLMMetaModel",  # Optional: Register in Model Registry
        signature=signature,  # Add the model signature
        input_example=input_example  # Optional: Example input for better documentation
    )

print(f"Model logged with run ID: {run.info.run_id}")


# COMMAND ----------

logged_model_uri = f"runs:/{run.info.run_id}/Model_pyfunc_model"
# logged_model_uri = f"runs:/{run_id}/Model_pyfunc_model"
loaded_model = mlflow.pyfunc.load_model(logged_model_uri)

print("Model Signature:")
print(loaded_model.metadata.signature)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model evaluation
# MAGIC For model evaluation we will:
# MAGIC 1. Define an evaluation dataset including the ground truth. This step is fundamental to have meaningful, actionable metrics.
# MAGIC 2. Run inference on the evaluation dataset using our model.
# MAGIC 3. Run mlflow evaluation metrics on the predictions infered by the model. For our particular use case we will use classification metrics. This metrics will be available in our experiment and will allow us to choose the best model out of several experiment.

# COMMAND ----------

import pandas as pd

# Define the evaluation dataset with vendor_name and binary predictions
eval_data = pd.DataFrame([
    {
        "vendor": "Vendor_1",
        "products": ["transformer", "wire", "insulation tape", "circuit breaker", "switchgear"],
        "ground_truth": 0,  # No anomaly expected
    },
    {
        "vendor": "Vendor_2",
        "products": ["electric drill", "mountain bike", "toolbox", "freight services", "hard hat"],
        "ground_truth": 1,  # Anomaly expected due to "mountain bike"
    },
    {
        "vendor": "Vendor_3",
        "products": ["lubricant", "hard hat", "freight services", "safety gloves", "toolbox"],
        "ground_truth": 0,  # No anomaly expected
    },
    {
        "vendor": "Vendor_4",
        "products": ["hammer", "screwdriver", "desk lamp", "extension cord", "flashlight"],
        "ground_truth": 1,  # Anomaly expected due to "desk lamp"
    },
    {
        "vendor": "Vendor_5",
        "products": ["cement", "bricks", "freight services", "transformer", "steel rods"],
        "ground_truth": 1,  # Anomaly expected due to "cement" and "bricks"
    },
    {
        "vendor": "Vendor_6",
        "products": ["power generator", "voltage regulator", "capacitor bank", "wires", "cooling fan"],
        "ground_truth": 0,  # No anomaly expected
    },
    {
        "vendor": "Vendor_7",
        "products": ["electrical conduit", "cable ties", "transformer", "wire", "PVC water pipe"],
        "ground_truth": 1,  # Anomaly expected due to "PVC water pipe"
    },
    {
        "vendor": "Vendor_8",
        "products": ["transformer", "hard hat", "fire extinguisher", "freight services", "lubricant"],
        "ground_truth": 0,  # No anomaly expected
    },
    {
        "vendor": "Vendor_9",
        "products": ["copper pipe", "plumbing tools", "circuit breaker", "wire", "freight services"],
        "ground_truth": 1,  # Anomaly expected due to "copper pipe" and "plumbing tools"
    },
    {
        "vendor": "Vendor_10",
        "products": ["safety harness", "insulation tape", "cable ties", "transformer", "steel rods"],
        "ground_truth": 0,  # No anomaly expected
    },
    {
        "vendor": "Vendor_11",
        "products": ["yoga mat", "safety gloves", "flashlight", "electric drill", "insulation tape"],
        "ground_truth": 1,  # Anomaly expected due to "yoga mat"
    },
    {
        "vendor": "Vendor_12",
        "products": ["solar panel", "battery storage system", "wire", "inverter", "circuit breaker"],
        "ground_truth": 0,  # No anomaly expected
    },
    {
        "vendor": "Vendor_13",
        "products": ["circuit breaker", "extension cord", "power strip", "fire extinguisher", "printer"],
        "ground_truth": 1,  # Anomaly expected due to "printer"
    },
    {
    "vendor": "Vendor_14",
    "products": ["steel rods", "bolts", "insulators", "transformer", "crane"],
    "ground_truth": 0,  # No anomaly expected; all items are typical for building a power tower
    },
    {
        "vendor": "Vendor_15",
        "products": ["electric motor", "cooling fan", "lubricant", "wires", "voltage regulator"],
        "ground_truth": 0,  # No anomaly expected
    }
])

# Display the DataFrame
eval_data


# COMMAND ----------

predictions = loaded_model.predict(eval_data)

# COMMAND ----------

predictions_df = pd.DataFrame(predictions)
predictions_df.head()

# COMMAND ----------

eval_data["model_prediction"] = predictions_df["model_prediction"]
eval_data["model_response"] = predictions_df["response"]

# COMMAND ----------

pd.set_option("display.max_colwidth", None)  
pd.set_option("display.max_columns", None) 
pd.set_option("display.max_rows", None)

eval_data

# COMMAND ----------

import mlflow

results = mlflow.evaluate(
    model=logged_model_uri,             # Model URI in MLflow
    data=eval_data,                     # Evaluation DataFrame
    targets="ground_truth",             # Ground truth labels
    predictions="model_prediction",     # Model's predicted labels
    model_type="classifier",            # Specify classification task
)


# COMMAND ----------

print(f"Evaluation metrics:\n{results.metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Register the model in UC
# MAGIC Once you are happy with the trained model, run the cell below to register it in UC. Change the model_name to your preferred name.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

model_name = "Model_LLMMetaModel" 
UC_MODEL_NAME = f"{catalog_name}.{schema_name}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_model_uri, name=UC_MODEL_NAME)

print(f"Registered model with name '{model_name}' and version '{uc_registered_model_info.version}'.")
