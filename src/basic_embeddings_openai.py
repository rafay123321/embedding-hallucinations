import os
from sentence_transformers import util
from openai import AzureOpenAI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv 


load_dotenv(override=True)
# Load the model
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),  
    azure_deployment=os.getenv("AZURE_DEPLOYMENT")
)
model_name = os.getenv("MODEL_NAME")
temperature = os.getenv("TEMPERATURE")
deployment_name = os.getenv("AZURE_DEPLOYMENT")

text = "American pizza is one of the nation's greatest cultural exports"
response = client.embeddings.create(
    input=text,
    model=deployment_name,
)

        
print(response.data)
    
