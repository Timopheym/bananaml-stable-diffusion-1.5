import os

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")
repo = 'runwayml/stable-diffusion-v1-5'

base_path = './base_model'
all_model_names = ['93612', '93640', '93715', '93744']


def get_model_url(model_name):
    return f"http://164.90.217.117/model_{model_name}.tar"


def get_filename(model_url):
    return model_url.split('/')[-1]


def get_model_path(filename):
    return filename.split('.')[0]
