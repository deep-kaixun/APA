from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import inspect
import torch.nn.functional as F
import random
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor
from  diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer
import torch.utils.checkpoint as checkpoint
from utils import ori_trans,sia
 


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def tensor2img(img):
    img=img.clone().cpu()
    transforms_=transforms.Compose([transforms.ToPILImage()])
    return transforms_(img)



def img2tensor(img):
    transforms_=transforms.Compose([transforms.ToTensor()])
    return transforms_(img)

def get_img(img_path, resolution=512):
    img = Image.open(img_path).convert("RGB")
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    img = transform(img)
    return img.unsqueeze(0)


class  AttackPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler,
                         safety_checker, feature_extractor, image_encoder, requires_safety_checker)

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0)
        latents = self.vae.encode(image.to(DEVICE))['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def get_text_embeddings(self, prompt, guidance_scale, neg_prompt, batch_size):
        DEVICE = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.cuda())[0]

        if guidance_scale > 1.:
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(
                unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat(
                [unconditional_embeddings, text_embeddings], dim=0)

        return text_embeddings

    @torch.no_grad()
    def ddim_inversion(self, latent, cond,inversion_step=None):
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps, desc="DDIM inversion")):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
                eps = self.unet(
                    latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                if inversion_step is not None and i==inversion_step:
                    print(t)
                    break
        return latent

    @torch.no_grad()
    def inverse(self, img_path, prompt, n_steps, neg_prompt='', guidance_scale=7.5, batch_size=1,inversion_step=None):
        self.scheduler.set_timesteps(n_steps)
        text_embeddings = self.get_text_embeddings(
            prompt, guidance_scale, neg_prompt, batch_size)
        img = get_img(img_path)
        img_noise = self.ddim_inversion(
            self.image2latent(img), text_embeddings,inversion_step)
        return img_noise
    
    def enimg2latent(self,img_path):
        img = get_img(img_path)
        return self.image2latent(img)
        
    
    def decode_la(self,latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5)
        return image

    
    
    @torch.no_grad()
    def cond_guidance(self, latents,
        timestep,
        noise_pred_original,classfier,label,ori_image_latent=None,use_checkpoint=False):
        noise_pred = noise_pred_original.clone()
        with torch.enable_grad():
            latents = latents.detach().requires_grad_()
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred.detach()) / alpha_prod_t ** (0.5)
            fac = torch.sqrt(beta_prod_t)
            sample = ori_image_latent * (fac) + pred_original_sample * (1 - fac)
            img=self.decode_la(sample)
            img=torch.clamp(img,0,1)
            img=F.interpolate(img,size=(self.image_size,self.image_size))
            out=classfier(img)
            loss=torch.nn.CrossEntropyLoss()(out, label)
            grads = torch.autograd.grad(loss, latents)[0]
            l1_grad = grads / torch.norm(grads, p=1)
            self.m=self.m+l1_grad.detach()
        if not use_checkpoint:
            noise_pred = noise_pred.detach() - torch.sqrt(beta_prod_t) * torch.sign(self.m)
            
            return noise_pred.detach()
        else:
            return torch.sqrt(beta_prod_t) * torch.sign(self.m)
    
    
    def vis(self,img):
        img=img[0].clone().cpu()
        transforms_=transforms.Compose([transforms.ToPILImage()])
        return transforms_(img).save('img2.png')

       

    def diffusion_augmentation(self,img0,la_list,p='ori'):
        img_size=img0.shape[-1]
        img_l=torch.zeros((len(la_list),3,img_size,img_size)).cuda()
        for i in range(len(la_list)):
            img=self.decode_la(la_list[i].detach()).detach()
            img=torch.clamp(img,0,1)
            img=F.interpolate(img.detach(),size=(img_size,img_size))
            img=(img0+img)/2
            if 'sia' in p:
                out=sia(img)
            else:
                out=ori_trans(img)
            img_l[i]=out
        return img_l
    


    @torch.no_grad()
    def attack_optimization(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        classfier=None,
        label=None,
        attack_config=None,
        use_noise_opt=True,
        use_da=True,
        index_cond=40,
        image_size=224,
        ori_latents=None,
        targeted=False,
        **kwargs,
    ):
        self.targeted=targeted
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False
        self.targetd=targeted
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        
       
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])


        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        self._num_timesteps = len(timesteps)
        
        #8. attack
        eps=attack_config['eps']
        alpha=attack_config['alpha']
        niters=attack_config['niters']
        la_0=latents.clone().detach()
        adv_latents=latents
        momentum=0
        self.image_size=image_size
        img_list=[]
        with tqdm(total=niters) as pbar:
            for ii in range(niters):
                la_list=[]
                self.m=0.
                latents=adv_latents.detach()
                #
                for i, t in enumerate(timesteps):   
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  
                    noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            return_dict=False,
                        )[0]
                    #9. noise opt
                    if use_noise_opt and i>=index_cond:
                        noise_pred_original = noise_pred.clone()
                        noise_pred = self.cond_guidance(latents, t, noise_pred_original, classfier, label, ori_latents)
                    # #vdi
                    if i>=index_cond and use_da:  
                        alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        beta_prod_t = 1 - alpha_prod_t
                        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                        la_list.append(pred_original_sample.detach())
                    if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                la_t=latents.detach()
                with torch.enable_grad():
                    la_t.requires_grad=True
                    img=self.decode_la(la_t)
                    img=torch.clamp(img,0,1)
                    img0=F.interpolate(img,size=(image_size,image_size))
                    if use_da:
                        img_l=self.diffusion_augmentation(img0,la_list)
                    else:
                        img_l=img0
                    out=classfier(img_l)
                    
                    if use_da:
                        reward=torch.nn.CrossEntropyLoss()(out, label.repeat_interleave(len(la_list)))
                    else:
                        reward=torch.nn.CrossEntropyLoss()(out, label)
                        
                    la_t_g=14.58*torch.autograd.grad(reward, la_t,
                                            retain_graph=False, create_graph=False)[0].detach()
                    l1_grad = la_t_g / torch.norm(la_t_g, p=1)
                    momentum = momentum + l1_grad
                    adv_latents = adv_latents + torch.sign(momentum) * alpha
                    noise = (adv_latents - la_0).clamp(-eps, eps)
                    adv_latents = la_0+ noise
                    pbar.update(1)
                    out=classfier(img0).detach()
                    pbar.set_postfix({'iters':ii,'attack_reward':reward.item(),'predicted':out.max(1)[1].item(),'label':label.item()})
        self.maybe_free_model_hooks()
        return img0

    
    @torch.no_grad()
    def vca(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        image_size=224,
        targeted=False,
        **kwargs,
    ):
        self.targeted=targeted
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False
        self.targetd=targeted
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        
       
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])


        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        self._num_timesteps = len(timesteps)
        self.image_size=image_size
        latents=latents.detach()+0.1*torch.randn_like(latents)
        for i, t in enumerate(timesteps):   
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  
            noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    return_dict=False,
                )[0]
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            img=self.decode_la(latents)
            img=torch.clamp(img,0,1)
            img0=F.interpolate(img,size=(image_size,image_size))
        self.maybe_free_model_hooks()
        return img0


    @torch.no_grad()
    def attack_optimization_checkpoint(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        classfier=None,
        label=None,
        attack_config=None,
        use_noise_opt=True,
        use_da=True,
        index_cond=40,
        image_size=224,
        ori_latents=None,
        inversion_step=10,
        targeted=False,
        **kwargs,
    ):
        self.targeted=targeted
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        
       
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])


        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        self._num_timesteps = len(timesteps)
        
        #8. attack
        eps=attack_config['eps']
        alpha=attack_config['alpha']
        niters=attack_config['niters']
        la_0=latents.clone().detach()
        adv_latents=latents
        momentum=0
        self.image_size=image_size
        with tqdm(total=niters) as pbar:
            for ii in range(niters):
                la_list=[]
                self.m=0.
                with torch.enable_grad():
                    adv_latents=adv_latents.requires_grad_()
                    latents=adv_latents
                    for i, t in enumerate(timesteps):
                        if i<len(timesteps)-inversion_step-1:
                            continue
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  
                        noise_pred=checkpoint.checkpoint(self.unet, latent_model_input, t, prompt_embeds, use_reentrant=False).sample
                        #9. noise opt
                        if use_noise_opt and i>=index_cond:
                            noise_pred_original = noise_pred.clone()
                            delta = self.cond_guidance(latents, t, noise_pred_original, classfier, label, ori_latents,use_checkpoint=True)
                            noise_pred = noise_pred - delta
                            
                        #vdi
                        if i>=index_cond and use_da:  
                            alpha_prod_t = self.scheduler.alphas_cumprod[t]
                            beta_prod_t = 1 - alpha_prod_t
                            pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                            la_list.append(pred_original_sample.detach())
                        if self.do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                        if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    la_t=latents
                    img=self.decode_la(la_t)
                    img=torch.clamp(img,0,1)
                    img0=F.interpolate(img,size=(image_size,image_size))
                    if use_da:
                        img_l=self.diffusion_augmentation(img0,la_list)
                    else:
                        img_l=img0
                    out=classfier(img_l)
                    if use_da:
                        reward=torch.nn.CrossEntropyLoss()(out, label.repeat_interleave(len(la_list)))-10*torch.nn.MSELoss()(ori_latents,la_t)
                    else:
                        reward=torch.nn.CrossEntropyLoss()(out, label)-10*torch.nn.MSELoss()(ori_latents,la_t)
                    
                    
                    la_t_g=torch.autograd.grad(reward, adv_latents,
                                            retain_graph=False, create_graph=False)[0].detach()
                l1_grad = la_t_g / torch.norm(la_t_g, p=1)
                momentum = momentum + l1_grad
                adv_latents = adv_latents + torch.sign(momentum) * alpha
                noise = (adv_latents - la_0).clamp(-eps, eps)
                adv_latents = la_0+ noise
                pbar.update(1)
                out=classfier(img0).detach()
                pbar.set_postfix({'iters':ii,'attack_reward':reward.item(),'predicted':out.max(1)[1].item(),'label':label.item()})
        self.maybe_free_model_hooks()
        return img0.detach()



    
