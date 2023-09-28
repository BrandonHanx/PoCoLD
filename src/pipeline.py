import inspect
import os
from typing import Callable, List, Optional, Union, Tuple

import torch
import torch.nn as nn
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    UNet2DModel,
)
from diffusers.models import DualTransformer2DModel
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import randn_tensor, logging
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from .model import ChannelMapper, ReferenceImageEncoder, MultiScaleReferenceImageEncoder


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DiffashionPipelineBase(DiffusionPipeline):
    def encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(self.device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(self.device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(self.device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        if self.vae is not None:
            latents = 1 / 0.18215 * latents
            image = self.vae.decode(latents).sample
        else:
            image = latents
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs


class PoseTransferPipeline(DiffashionPipelineBase):
    def __init__(
        self,
        vae: AutoencoderKL,
        rie: Union[ReferenceImageEncoder, MultiScaleReferenceImageEncoder],
        channel_mapper: ChannelMapper,
        unet: Union[UNet2DModel, UNet2DConditionModel],
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            rie=rie,
            unet=unet,
            channel_mapper=channel_mapper,
            scheduler=scheduler,
        )
        if vae is not None:
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        else:
            self.vae_scale_factor = 1

    @torch.no_grad()
    def __call__(
        self,
        ref: torch.FloatTensor,  # TODO: pass PIL here
        cat: torch.FloatTensor,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        cat_guidance_scale: Optional[float] = 1.0,
        ref_guidance_scale: Optional[float] = 1.0,
        cfg_type: Optional[str] = "cat_ref",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        eta: Optional[float] = 0.0,
        constraint_maps: Optional[float] = None,
        cfg_decay: bool = False,
        end_cfg: float = 2.0,
        source: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:

        assert cfg_type in [
            "no",
            "cat_only",
            "cat_ref",
            "ref_cat",
            "dis",
            "and",
        ]

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        # get ref embeddings
        if self.vae is not None:
            ref = self.vae.encode(ref.to(self.device)).latent_dist.sample()
            ref = ref * 0.18215
        # In case rie doesn't need time embeddings, we calculate it here
        if self.rie.config.use_time_emb:
            ref_embeddings = None
        else:
            ref_embeddings = self.rie(ref)
        if source is not None:
            source_latents = (
                self.vae.encode(source.to(self.device)).latent_dist.sample() * 0.18215
            )

        # get the initial random noise unless the user supplied it
        if ref.dim() > 3:
            batch_size = len(ref)
        else:
            batch_size = 1

        # get unconditional embeddings for classifier free guidance
        if ref_guidance_scale != 1.0:
            ref_embeddings_uncond = None
            # ref_embeddings_uncond = self.rie(torch.zeros_like(ref).to(self.device))

        # TODO: add channel calculation
        if self.vae is not None:
            latents_shape = (
                batch_size,
                4,
                # self.unet.in_channels // 2,
                height // 8,
                width // 8,
            )
        else:
            latents_shape = (
                batch_size,
                self.unet.in_channels // 2,
                height,
                width,
            )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                latents_shape,
                generator=generator,
                device=self.device,
                dtype=self.unet.dtype,
            )
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
                )
        latents = latents.to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        attns = []

        for t in self.progress_bar(self.scheduler.timesteps):
            if ref_embeddings is None:
                ref_embeddings = self.rie(ref, t.to(self.device))
            if cfg_type == "no":
                latents_input = self.channel_mapper(latents, cat)
                latents_input = self.scheduler.scale_model_input(latents_input, t)
                noise_pred, p = self.unet(
                    latents_input,
                    t,
                    encoder_hidden_states=ref_embeddings,
                    constraint_maps=constraint_maps,
                )
                noise_pred = noise_pred.sample
                attns.append(p)
            elif cfg_type == "cat_only":
                latents_input = self.channel_mapper(latents, cat)
                latents_input_uncond = self.channel_mapper(latents, cat * 0.0)
                noise_null = self.unet(
                    latents_input_uncond,
                    t,
                    encoder_hidden_states=ref_embeddings_uncond,
                    constraint_maps=None,
                ).sample
                noise_cat = self.unet(
                    latents_input,
                    t,
                    encoder_hidden_states=ref_embeddings_uncond,
                    constraint_maps=None,
                ).sample
                if cfg_decay:
                    noise_pred = noise_null + max(
                        cat_guidance_scale * t.cpu().numpy() / 1000, 2
                    ) * (noise_cat - noise_null)
                else:
                    noise_pred = noise_null + cat_guidance_scale * (
                        noise_cat - noise_null
                    )
            elif cfg_type == "cat_ref":
                latents_input = self.channel_mapper(latents, cat)
                latents_input_uncond = self.channel_mapper(latents, cat * 0.0)
                noise_null = self.unet(
                    latents_input_uncond,
                    t,
                    encoder_hidden_states=ref_embeddings_uncond,
                    constraint_maps=None,
                ).sample
                noise_cat = self.unet(
                    latents_input,
                    t,
                    encoder_hidden_states=ref_embeddings_uncond,
                    constraint_maps=None,
                ).sample
                noise_all = self.unet(
                    latents_input,
                    t,
                    encoder_hidden_states=ref_embeddings,
                    constraint_maps=constraint_maps,
                ).sample
                if cfg_decay:
                    noise_pred = (
                        noise_null
                        + max(ref_guidance_scale * t.cpu().numpy() / 1000, end_cfg)
                        * (noise_all - noise_cat)
                        + max(cat_guidance_scale * t.cpu().numpy() / 1000, end_cfg)
                        * (noise_cat - noise_null)
                    )
                else:
                    noise_pred = (
                        noise_null
                        + ref_guidance_scale * (noise_all - noise_cat)
                        + cat_guidance_scale * (noise_cat - noise_null)
                    )
            elif cfg_type == "ref_cat":
                latents_input = self.channel_mapper(latents, cat)
                latents_input_uncond = self.channel_mapper(latents, cat * 0.0)
                noise_null = self.unet(
                    latents_input_uncond,
                    t,
                    encoder_hidden_states=ref_embeddings_uncond,
                    constraint_maps=None,
                ).sample
                noise_ref = self.unet(
                    latents_input_uncond,
                    t,
                    encoder_hidden_states=ref_embeddings,
                    constraint_maps=None,
                ).sample
                noise_all = self.unet(
                    latents_input,
                    t,
                    encoder_hidden_states=ref_embeddings,
                    constraint_maps=constraint_maps,
                ).sample
                noise_pred = (
                    noise_null
                    + ref_guidance_scale * (noise_ref - noise_null)
                    + cat_guidance_scale * (noise_all - noise_ref)
                )
            elif cfg_type == "dis":
                latents_input = self.channel_mapper(latents, cat)
                latents_input_uncond = self.channel_mapper(latents, cat * 0.0)
                noise_null = self.unet(
                    latents_input_uncond,
                    t,
                    encoder_hidden_states=ref_embeddings_uncond,
                    constraint_maps=None,
                ).sample
                noise_ref = self.unet(
                    latents_input_uncond,
                    t,
                    encoder_hidden_states=ref_embeddings,
                    constraint_maps=None,
                ).sample
                noise_cat = self.unet(
                    latents_input,
                    t,
                    encoder_hidden_states=ref_embeddings_uncond,
                    constraint_maps=None,
                ).sample
                noise_pred = (
                    noise_null
                    + ref_guidance_scale * (noise_ref - noise_null)
                    + cat_guidance_scale * (noise_cat - noise_null)
                )
            elif cfg_type == "and":
                latents_input = self.channel_mapper(latents, cat)
                latents_input_uncond = self.channel_mapper(latents, cat * 0.0)
                noise_null = self.unet(
                    latents_input_uncond,
                    t,
                    encoder_hidden_states=ref_embeddings_uncond,
                    constraint_maps=None,
                ).sample
                noise_all = self.unet(
                    latents_input,
                    t,
                    encoder_hidden_states=ref_embeddings,
                    constraint_maps=constraint_maps,
                ).sample
                noise_pred = noise_null + ref_guidance_scale * (noise_all - noise_null)

            if source is not None:
                pred_latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                noise = torch.randn_like(source_latents)
                latents = self.scheduler.add_noise(source_latents, noise, t)
                latents[mask] = pred_latents[mask]
            else:
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

        # scale and decode the image latents with vae
        image = self.decode_latents(latents)
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class UnconditionalPipeline(DiffashionPipelineBase):
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: Union[UNet2DModel, UNet2DConditionModel],
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
        )
        if vae is not None:
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        else:
            self.vae_scale_factor = 1

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        eta: Optional[float] = 0.0,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        latents_shape = (
            batch_size,
            4,
            # self.unet.in_channels // 2,
            height // 8,
            width // 8,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                latents_shape,
                generator=generator,
                device=self.device,
                dtype=self.unet.dtype,
            )
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
                )
        latents = latents.to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        for t in self.progress_bar(self.scheduler.timesteps):
            noise_pred = self.unet(
                latents,
                t,
            ).sample

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        # scale and decode the image latents with vae
        image = self.decode_latents(latents)
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
