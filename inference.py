import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from diffusers import DDPMScheduler, DDIMScheduler
from PIL import Image, ImageOps
from src.model import (
    ChannelMapper,
    MultiScaleReferenceImageEncoder,
    UNet2DDoubleAttentionConditionModel,
)
from src.pipeline import PoseTransferPipeline
from src.transform import build_transforms, build_dual_transforms
from tqdm import tqdm

path_name = "final"
path = "models/final/"
channel_mapper = ChannelMapper.from_pretrained(
    f"{path}/channel_mapper/",
    torch_dtype=torch.float16,
)
rie = MultiScaleReferenceImageEncoder.from_pretrained(
    f"{path}/rie/",
    torch_dtype=torch.float16,
    # use_cls_token=False,
    # use_time_emb=True,
)
unet = UNet2DDoubleAttentionConditionModel.from_pretrained(
    f"{path}/unet/",
    attn_heads_nums=8,
    torch_dtype=torch.float16,
)
pipe = PoseTransferPipeline.from_pretrained(
    path,
    unet=unet,
    rie=rie,
    channel_mapper=channel_mapper,
    torch_dtype=torch.float16,
    # scheduler=ddpm,
).to("cuda")

val_dataset = load_from_disk("datasets/DeepFashionUnify/")["validation"]

height = 512
width = 384
val_transforms = build_dual_transforms(height, width, 25, 8, is_train=False)


def preprocess_val(examples):
    examples["cat_values"] = []
    examples["ref_values"] = []
    examples["constraint_maps"] = []
    for image, ref in zip(examples["to_image"], examples["from_image"]):
        cat = image.replace("img_highres", "img_dp").replace(".jpg", ".png")
        ref_cat = ref.replace("img_highres", "img_dp").replace(".jpg", ".png")
        ref = Image.open(ref)
        cat = Image.open(cat)
        ref_cat = Image.open(ref_cat)
        ref, cat = val_transforms(ref.convert("RGB"), cat)
        _, ref_cat = val_transforms(None, ref_cat)
        constraint_map = torch.cat(
            [
                torch.argmax(cat, dim=0, keepdim=True),
                torch.argmax(ref_cat, dim=0, keepdim=True),
            ]
        )
        examples["cat_values"].append(cat)
        examples["ref_values"].append(ref)
        examples["constraint_maps"].append(constraint_map)
    return examples


def collate_fn(examples):
    cat_values = torch.stack([example["cat_values"] for example in examples])
    cat_values = cat_values.to(memory_format=torch.contiguous_format).float()
    ref_values = torch.stack([example["ref_values"] for example in examples])
    ref_values = ref_values.to(memory_format=torch.contiguous_format).float()
    constraint_maps = torch.stack([example["constraint_maps"] for example in examples])
    constraint_maps = constraint_maps.to(memory_format=torch.contiguous_format).float()
    save_paths = [
        example["org_from_image"].replace(".jpg", "_2_")
        + example["org_to_image"].replace(".jpg", "_vis.png")
        for example in examples
    ]
    return {
        "cat_values": cat_values,
        "ref_values": ref_values,
        "save_paths": save_paths,
        "constraint_maps": constraint_maps,
    }


val_dataset = val_dataset.with_transform(preprocess_val)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, shuffle=False, collate_fn=collate_fn, batch_size=32
)

for batch in tqdm(val_dataloader):
    gen_images = pipe(
        ref=batch["ref_values"].half().cuda(),
        cat=batch["cat_values"].half().cuda(),
        height=height,
        width=width,
        ref_guidance_scale=5,
        cat_guidance_scale=5,
        num_inference_steps=50,
        cfg_decay=True,
        end_cfg=3,
        generator=torch.Generator(device="cuda").manual_seed(42),
        # constraint_maps=batch["constraint_maps"].half().cuda(),
    ).images
    for gen, path in zip(gen_images, batch["save_paths"]):
        path = os.path.join(f"gens/{path_name}/", path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        gen.save(path)
    # break
