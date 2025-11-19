
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import diffusers
import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import csv
import time
import json
from PIL import Image
# Preprocessing the datasets.
train_transforms = transforms.Compose(
        [
            transforms.Resize(
                512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

def tokenize_captions(caption,tokenizer, is_train=True):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


def main(train_cfg,data_cfg,out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
    )

    if train_cfg['seed'] is not None:
        set_seed(train_cfg['seed'])
    pretrained_model_name_or_path=train_cfg['pretrained_model_name_or_path']
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32

    # Freeze the unet parameters before adding adapters
    for param in unet.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=train_cfg['rank'],
        lora_alpha=train_cfg['rank'],
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    unet.add_adapter(unet_lora_config)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers,
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )


    # Scheduler and math around the number of training steps.
    max_train_steps = train_cfg['n_epochs']

    lr_scheduler = get_scheduler(
        'constant',
        optimizer=optimizer
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer,  lr_scheduler = accelerator.prepare(
        unet, optimizer, lr_scheduler
    )


    num_train_epochs = max_train_steps

    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    image_path=data_cfg['image_path']
    caption=data_cfg['class']
    batch_data={}
    image = Image.open(image_path).convert("RGB")
    batch_data['pixel_values'] = train_transforms(image).to(accelerator.device)
    batch_data['pixel_values'] = batch_data['pixel_values'].unsqueeze(0)
    batch_data['input_ids'] = tokenize_captions(caption,tokenizer).to(accelerator.device)
    batch_data['input_ids'] = batch_data['input_ids'].unsqueeze(0)
    global_step=0
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        with accelerator.accumulate(unet):
            latents = vae.encode(batch_data["pixel_values"].to(
                    dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            if train_cfg['noise_offset']:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += train_cfg['noise_offset'] * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch_data["input_ids"])[0]

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(
                    latents, noise, timesteps)

            # Predict the noise residual and compute loss
            model_pred = unet(noisy_latents, timesteps,
                                  encoder_hidden_states).sample

            loss = F.mse_loss(model_pred.float(),
                                target.float(), reduction="mean")

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(
                loss.repeat(train_cfg['train_batch_size'])).mean()
            train_loss += avg_loss.item() / train_cfg['gradient_accumulation_steps']

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = lora_layers
                accelerator.clip_grad_norm_(
                    params_to_clip, train_cfg['max_grad_norm'])
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unwrapped_unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=out_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
            weight_name=str(data_cfg['id']) + '.safetensor'
        )

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,default='data.json')
    parser.add_argument('--output_path', type=str,default='ckpt_2025')
    parser.add_argument('--num', type=int,default=5)
    args = parser.parse_args()
    with open(args.data_path) as f:
        data=json.load(f)
    cfg={
    "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
    "seed":0,
    "rank": 8,
    "n_epochs": 200,
    "checkpointing_steps": 500,
    "noise_offset": 0.1,
    "max_grad_norm": 1.0,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    }
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    for i,jdata in enumerate(data):
        if str(jdata['id'])+'.safetensor' not in os.listdir(args.output_path):
            main(train_cfg=cfg,data_cfg=jdata,out_dir=args.output_path)
