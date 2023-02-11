import os

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

model_url = f"http://164.90.217.117/model_{MODEL_NAME}.tar"
filename = model_url.split('/')[-1]
model_path = filename.split('.')[0]
repo = 'runwayml/stable-diffusion-v1-5'
base_path = './base_model'
