import os
import torch
import urllib.request
from config import model_url, filename
from diffusers import StableDiffusionPipeline, DDIMScheduler
import tarfile


def untar(filename):
    tar = tarfile.open(filename, 'r:')
    tar.extractall(filename.split('.')[0])
    tar.close()



def download_model():
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    urllib.request.urlretrieve(model_url, filename)
    untar(filename)
    model_path = filename.split('.')[0]
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)
    pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, safety_checker=None,
                                                   torch_dtype=torch.float16).to("cuda")
    
if __name__ == "__main__":
    download_model()
