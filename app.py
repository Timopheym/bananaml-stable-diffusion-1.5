import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from config import base_path

import base64
from io import BytesIO
from logging import getLogger

logger = getLogger(__name__)

def init():
    global model
    # scheduler = DPMSolverMultistepScheduler.from_pretrained(filename, subfolder="scheduler")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)
    logger.warning(f"Loading model from {base_path}")
    model = StableDiffusionPipeline.from_pretrained(base_path, scheduler=scheduler, safety_checker=None,
                                                   torch_dtype=torch.float16).to("cuda")


def inference(model_inputs:dict):
    global model

    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 768)
    width = model_inputs.get('width', 768)
    steps = model_inputs.get('steps', 20)
    guidance_scale = model_inputs.get('guidance_scale', 9)
    seed = model_inputs.get('seed', None)

    logger.warning(f"Received prompt: {prompt}")

    if not prompt: return {'message': 'No prompt was provided'}

    generator = None
    if seed: generator = torch.Generator("cuda").manual_seed(seed)
    
    with autocast("cuda"):
        image = model(prompt, guidance_scale=guidance_scale, height=height, width=width, num_inference_steps=steps, generator=generator).images[0]
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {'image_base64': image_base64}
