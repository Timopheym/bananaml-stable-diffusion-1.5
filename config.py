model_url = "http://164.90.217.117/model_93612.tar"
filename = model_url.split('/')[-1]
model_path = filename.split('.')[0]
repo = 'jcplus/stable-diffusion-v1-5'
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
base_path = './base_model'