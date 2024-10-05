import streamlit as st
import time
from PIL import Image
import os
import boto3
from transformers import pipeline
import torch




local_path = 'tiny_bert_sentiment_analysis_downloaded'
s3_prefix = 'ml_model/tiny_bert_sentiment_analysis'
bucket_name = 'prjsentimentanalysis'

s3 = boto3.client('s3')

def download_dir(local_path,s3_prefix):
    os.makedirs(local_path,exist_ok= True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket = bucket_name,Prefix =s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']
                
                local_file = os.path.join(local_path,os.path.relpath(s3_key,s3_prefix))
                                          
                s3.download_file(bucket_name,s3_key,local_file)

st.title('Machine Learning Model Deployment at Streamlit Server')
button = st.button('Download Model')
if button:
    with st.spinner('Downloading.. Please Wait'):
        download_dir(local_path,s3_prefix)

text = st.text_area('Enter you Review','Type...')
predict = st.button('Predict')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
classifier = pipeline('text-classification', model = 'tiny_bert_sentiment_analysis',device = device)
if predict:
    with st.spinner('Predicting....'):
        output = classifier(text)
        st.markdown(f"Movie Review is ___{output[0]['label']}___, with ___{round(output[0]['score']*100,0)}%___ probability")
# st.info(output)
