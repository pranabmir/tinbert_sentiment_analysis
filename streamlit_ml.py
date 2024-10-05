import streamlit as st
import os
import boto3
from transformers import pipeline
import torch

# Constants for S3 bucket and model path
local_path = 'tiny_bert_sentiment_analysis'
s3_prefix = 'ml_model/tiny_bert_sentiment_analysis'
bucket_name = 'prjsentimentanalysis'

# Initialize S3 client
s3 = boto3.client('s3')

def download_dir(local_path, s3_prefix):
    """
    Download all files from an S3 directory (prefix) to a local directory.
    """
    os.makedirs(local_path, exist_ok=True)  # Ensure local directory exists
    paginator = s3.get_paginator('list_objects_v2')

    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for obj in result['Contents']:
                s3_key = obj['Key']
                
                # Skip if the key is a directory
                if s3_key.endswith('/'):
                    continue
                
                # Construct the local file path
                relative_path = os.path.relpath(s3_key, s3_prefix)
                local_file = os.path.join(local_path, relative_path)
                
                # Ensure local directory exists for this file
                local_dir = os.path.dirname(local_file)
                os.makedirs(local_dir, exist_ok=True)

                # Download the file
                s3.download_file(bucket_name, s3_key, local_file)
                st.write(f"Downloaded: {local_file}")

# Streamlit UI
st.title('Machine Learning Model Deployment at Streamlit Server')

button = st.button('Download Model')
if button:
    with st.spinner('Downloading.. Please Wait'):
        download_dir(local_path, s3_prefix)
        st.success("Model downloaded successfully.")

# Input area for text
text = st.text_area('Enter your Review', 'Type...')

# Button to predict
predict = st.button('Predict')

# Load the model only if it has been downloaded
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if os.path.exists(local_path) and os.listdir(local_path):  # Ensure model files exist locally
    classifier = pipeline('text-classification', model=local_path, device=device)
else:
    classifier = None
    st.error('Model not found. Please download the model first.')

# Prediction logic
if predict:
    if classifier is not None:
        with st.spinner('Predicting....'):
            output = classifier(text)
            st.markdown(f"Movie Review is ___{output[0]['label']}___, with ___{round(output[0]['score']*100,0)}%___ probability")
    else:
        st.error('Model has not been downloaded. Please click the "Download Model" button first.')
