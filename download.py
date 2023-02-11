import os
import torch
import urllib.request
from config import model_url, filename
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


def download_model():
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    urllib.request.urlretrieve(model_url, filename)
    # scheduler = DPMSolverMultistepScheduler.from_pretrained(filename, subfolder="scheduler")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)
    model = DiffusionPipeline.from_pretrained(filename,
                                              torch_dtype=torch.float16,
                                              revision="fp16",
                                              scheduler=scheduler,
                                              use_auth_token=HF_AUTH_TOKEN,
                                              safety_checker=None)
    
if __name__ == "__main__":
    download_model()
