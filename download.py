import os
import torch
import urllib.request
from config import model_url, filename, model_path, HF_AUTH_TOKEN, base_path, repo
from diffusers import StableDiffusionPipeline, DDIMScheduler, DiffusionPipeline, DPMSolverMultistepScheduler
import shutil
import tarfile
from logging import getLogger

logger = getLogger(__name__)


def untar(filename):
    tar = tarfile.open(filename, 'r:')
    tar.extractall(filename.split('.')[0])
    tar.close()


def copy_files(src, dst):
    folders = os.listdir(src)
    for folder in folders:
        # Check if directory
        full_src_path = os.path.join(src, folder)
        if os.path.isdir(full_src_path):
            full_dst_path = os.path.join(dst, folder)

            files = os.listdir(full_src_path)
            for file in files:
                full_src_file_path = os.path.join(full_src_path, file)
                full_dst_file_path = os.path.join(full_dst_path, file)
                shutil.copy(full_src_file_path, full_dst_file_path)


def download_model():
    # Download model
    logger.info(f"Downloading model from {repo}")
    scheduler_for_initial_download = DPMSolverMultistepScheduler.from_pretrained(repo, subfolder="scheduler")
    model_for_initial_download = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16,
                                                                   scheduler=scheduler_for_initial_download,
                                                                   use_auth_token=HF_AUTH_TOKEN, safety_checker=None)
    # Create model folder
    logger.info(f"Creating model folder {base_path}")
    model_for_initial_download.save_pretrained(base_path)
    # Download pretrained model
    logger.info(f"Downloading pretrained model from {model_url}")
    urllib.request.urlretrieve(model_url, filename)
    logger.info(f"Unzipping model {filename}")
    untar(filename)

    logger.info(f"Copying files from {model_path} to {base_path}")
    copy_files(model_path, base_path)


if __name__ == "__main__":
    download_model()
