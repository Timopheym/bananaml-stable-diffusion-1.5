import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler, StableDiffusionLatentUpscalePipeline
from config import base_path, all_model_names

import base64
from io import BytesIO
from logging import getLogger

logger = getLogger(__name__)


def init():
    global models
    global upscaler
    models = {}
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler",
                                                                    torch_dtype=torch.float16)
    upscaler.to("cuda")

    for model_name in all_model_names:
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        logger.warning(f"Loading model from {base_path}")
        model = StableDiffusionPipeline.from_pretrained(base_path, scheduler=scheduler, safety_checker=None,
                                                        torch_dtype=torch.float16).to("cuda")
        models[model_name] = model


def inference(model_inputs: dict):
    global models

    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 768)
    width = model_inputs.get('width', 768)
    steps = model_inputs.get('steps', 20)
    guidance_scale = model_inputs.get('guidance_scale', 9)
    seed = model_inputs.get('seed', None)
    model_name = model_inputs.get('model_name', None)

    logger.warning(f"Received prompt: {prompt}")

    if not prompt: return {'message': 'No prompt was provided'}
    if not model_name: return {'message': 'No model name was provided'}

    model = models.get(model_name, None)

    if not model: return {'message': 'Model not found'}

    generator = None
    if seed: generator = torch.Generator("cuda").manual_seed(seed)

    with autocast("cuda"):
        low_res_latents = model(prompt, guidance_scale=guidance_scale, height=height, width=width,
                                num_inference_steps=steps,
                                generator=generator).images

    upscaled_images = upscaler(
        prompt=prompt,
        image=low_res_latents,
        num_inference_steps=20,
        guidance_scale=0,
        generator=generator,
    ).images

    images_base64 = []

    for upscaled_image in upscaled_images:
        buffered = BytesIO()
        upscaled_image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        images_base64.append(image_base64)

    return {'images_base64': images_base64}
