import argparse
import copy
import itertools
import logging
import math
import os
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, load_from_disk
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
)
from diffusers.configuration_utils import FrozenDict
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from src.model import (
    ChannelMapper,
    EMAModel,
    MultiScaleReferenceImageEncoder,
    UNet2DDoubleAttentionConditionModel,
)
from src.pipeline import PoseTransferPipeline
from src.transform import build_dual_transforms, build_transforms
from torchvision import transforms
from tqdm.auto import tqdm

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="to_image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--ref_column",
        type=str,
        default="from_image",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--cat_column",
        type=str,
        default="to_densepose",
        help="The column of the dataset containing semantic maps.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=352,
    )
    parser.add_argument(
        "--cat_in_channels",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--cat_out_channels",
        type=int,
        default=4,
    )
    parser.add_argument("--do_cfg", action="store_true", default=False)
    parser.add_argument(
        "--cfg_value",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_x",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--use_constraint_penalty",
        action="store_true",
    )
    parser.add_argument(
        "--use_image",
        action="store_true",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--show_steps",
        type=int,
        default=1000,
        help="Number of steps to show val generation results",
    )
    parser.add_argument(
        "--save_steps", type=int, default=10000, help="Number of steps to save ckpt"
    )
    parser.add_argument(
        "--load_from_disk",
        action="store_true",
        default=False,
        help="Whether load the dataset from Arrow",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        kwargs_handlers=[ddp_kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token
                )
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load models and create wrapper
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
    )
    if args.from_scratch:
        unet = UNet2DDoubleAttentionConditionModel(
            in_channels=4 + args.cat_out_channels,
            sample_size=64,
            use_constraint_penalty=args.use_constraint_penalty,
            # norm_eps=1e-8,
            attn_heads_nums=8,
        )
        channel_mapper = ChannelMapper(
            args.cat_in_channels, args.cat_out_channels, concat=True
        )
        rie = MultiScaleReferenceImageEncoder(
            # use_time_emb=True,
            # norm_eps=1e-8,
        )
    else:
        unet = UNet2DDoubleAttentionConditionModel.from_pretrained(
            f"{args.pretrained_model_name_or_path}/unet/",
        )
        channel_mapper = ChannelMapper.from_pretrained(
            f"{args.pretrained_model_name_or_path}/channel_mapper/"
        )
        rie = MultiScaleReferenceImageEncoder.from_pretrained(
            f"{args.pretrained_model_name_or_path}/rie/",
        )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = copy.deepcopy(unet)
        ema_unet = EMAModel(ema_unet.parameters())
        ema_rie = copy.deepcopy(rie)
        ema_rie = EMAModel(ema_rie.parameters())
        ema_channel_mapper = copy.deepcopy(channel_mapper)
        ema_channel_mapper = EMAModel(ema_channel_mapper.parameters())

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params = [
        {"params": unet.parameters()},
        {"params": channel_mapper.parameters(), "lr": args.learning_rate * args.lr_x},
        {"params": rie.parameters(), "lr": args.learning_rate * args.lr_x},
    ]

    optimizer = optimizer_cls(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.load_from_disk:
            print(f"Loading dataset from {args.train_data_dir}...")
            dataset = load_from_disk(args.train_data_dir)
        else:
            # data_files = {}
            # if args.train_data_dir is not None:
            #     data_files["train"] = os.path.join(args.train_data_dir, "**")
            dataset = load_dataset(
                "imagefolder",
                # data_files=data_files,
                data_dir=args.train_data_dir,
                cache_dir=args.cache_dir,
            )
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    image_column = args.image_column
    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
        )
    ref_column = args.ref_column
    if ref_column not in column_names:
        raise ValueError(
            f"--ref_column' value '{args.ref_column}' needs to be one of: {', '.join(column_names)}"
        )
    cat_column = args.cat_column
    if cat_column not in column_names:
        raise ValueError(
            f"--cat_column' value '{args.cat_column}' needs to be one of: {', '.join(column_names)}"
        )

    ratio = 1 if args.use_image else 2 ** (len(vae.config.block_out_channels) - 1)
    train_transforms = build_dual_transforms(
        args.height, args.width, args.cat_in_channels, ratio, is_train=True
    )
    val_transforms = build_dual_transforms(
        args.height, args.width, args.cat_in_channels, ratio, is_train=False
    )
    ref_train_transforms = build_transforms(args.height, args.width, is_train=True)
    ref_val_transforms = build_transforms(args.height, args.width, is_train=False)

    def preprocess_train(examples):
        examples["pixel_values"] = []
        examples["cat_values"] = []
        examples["ref_values"] = []
        if args.use_constraint_penalty:
            examples["constraint_maps"] = []
            for image, cat, ref, ref_cat in zip(
                examples[image_column],
                examples[cat_column],
                examples[ref_column],
                examples["from_densepose"],
            ):
                image, cat = train_transforms(image.convert("RGB"), cat)
                ref, ref_cat = train_transforms(ref.convert("RGB"), ref_cat)
                examples["pixel_values"].append(image)
                examples["cat_values"].append(cat)
                examples["ref_values"].append(ref)
                # TODO: support case that has different resolutions
                constraint_maps = torch.cat(
                    [
                        torch.argmax(cat, dim=0, keepdim=True),
                        torch.argmax(ref_cat, dim=0, keepdim=True),
                    ]
                )
                examples["constraint_maps"].append(constraint_maps)
        else:
            for image, cat, ref in zip(
                examples[image_column], examples[cat_column], examples[ref_column]
            ):
                image, cat = train_transforms(image.convert("RGB"), cat)
                ref = ref_train_transforms(ref.convert("RGB"))
                examples["pixel_values"].append(image)
                examples["cat_values"].append(cat)
                examples["ref_values"].append(ref)

        return examples

    def preprocess_val(examples):
        examples["pixel_values"] = []
        examples["cat_values"] = []
        examples["ref_values"] = []
        if args.use_constraint_penalty:
            examples["constraint_maps"] = []
            for image, cat, ref, ref_cat in zip(
                examples[image_column],
                examples[cat_column],
                examples[ref_column],
                examples["from_densepose"],
            ):
                image, cat = val_transforms(image.convert("RGB"), cat)
                ref, ref_cat = val_transforms(ref.convert("RGB"), ref_cat)
                examples["pixel_values"].append(image)
                examples["cat_values"].append(cat)
                examples["ref_values"].append(ref)
                constraint_maps = torch.cat(
                    [
                        torch.argmax(cat, dim=0, keepdim=True),
                        torch.argmax(ref_cat, dim=0, keepdim=True),
                    ]
                )
                examples["constraint_maps"].append(constraint_maps)
        else:
            for image, cat, ref in zip(
                examples[image_column], examples[cat_column], examples[ref_column]
            ):
                image, cat = val_transforms(image.convert("RGB"), cat)
                ref = ref_val_transforms(ref.convert("RGB"))
                examples["pixel_values"].append(image)
                examples["cat_values"].append(cat)
                examples["ref_values"].append(ref)

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"]
                .shuffle(seed=args.seed)
                .select(range(args.max_train_samples))
            )
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)
        val_dataset = dataset["validation"].with_transform(preprocess_val)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        cat_values = torch.stack([example["cat_values"] for example in examples])
        cat_values = cat_values.to(memory_format=torch.contiguous_format).float()
        ref_values = torch.stack([example["ref_values"] for example in examples])
        ref_values = ref_values.to(memory_format=torch.contiguous_format).float()
        if args.use_constraint_penalty:
            constraint_maps = torch.stack(
                [example["constraint_maps"] for example in examples]
            )
            constraint_maps = constraint_maps.to(
                memory_format=torch.contiguous_format
            ).float()
            return {
                "pixel_values": pixel_values,
                "cat_values": cat_values,
                "ref_values": ref_values,
                "constraint_maps": constraint_maps,
            }
        return {
            "pixel_values": pixel_values,
            "cat_values": cat_values,
            "ref_values": ref_values,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=8,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, shuffle=True, collate_fn=collate_fn, batch_size=4
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps
        * args.gradient_accumulation_steps
        * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        * args.gradient_accumulation_steps
        * accelerator.num_processes,
    )

    (
        unet,
        channel_mapper,
        rie,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        unet,
        channel_mapper,
        rie,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )

    if args.use_ema:
        accelerator.register_for_checkpointing(ema_unet)
        accelerator.register_for_checkpointing(ema_rie)
        accelerator.register_for_checkpointing(ema_channel_mapper)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema:
        ema_unet.to(accelerator.device)
        ema_rie.to(accelerator.device)
        ema_channel_mapper.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("semantic-fine-tune", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    for epoch in range(args.num_train_epochs):
        unet.train()
        channel_mapper.train()
        rie.train()
        train_loss = 0.0
        train_attn_penalty = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet) as _, accelerator.accumulate(
                channel_mapper
            ) as _, accelerator.accumulate(rie) as _:
                if args.do_cfg:
                    # batch["ref_values"] = None
                    rand = torch.rand(1).cuda()
                    rand = accelerator.gather(rand)[0]
                    if rand < args.cfg_value:
                        batch["cat_values"] = batch["cat_values"] * 0.0
                    elif args.cfg_value <= rand < args.cfg_value * 2:
                        batch["ref_values"] = None

                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(weight_dtype)
                ).latent_dist.sample()
                latents = latents * 0.18215

                if batch["ref_values"] is not None:
                    ref_latents = vae.encode(
                        batch["ref_values"].to(weight_dtype)
                    ).latent_dist.sample()
                    ref_latents = ref_latents * 0.18215
                else:
                    ref_latents = None

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                rie_timesteps = timesteps
                # FIXME: add option here
                # rie_timesteps = None
                encoder_hidden_states = (
                    rie(ref_latents, rie_timesteps) if ref_latents is not None else None
                )

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Predict the noise residual and compute loss
                noisy_latents = channel_mapper(
                    noisy_latents, batch["cat_values"].to(weight_dtype)
                )
                constraint_maps = (
                    batch["constraint_maps"].to(weight_dtype)
                    if args.use_constraint_penalty
                    else None
                )
                if args.do_cfg and rand < args.cfg_value * 2:
                    constraint_maps = None
                model_pred, attn_penalty = unet(
                    noisy_latents, timesteps, encoder_hidden_states, constraint_maps
                )
                loss = F.mse_loss(
                    model_pred.sample.float(), target.float(), reduction="mean"
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                if isinstance(attn_penalty, int):
                    train_attn_penalty = 0
                else:
                    loss = loss + attn_penalty
                    avg_attn_penalty = accelerator.gather(
                        attn_penalty.repeat(args.train_batch_size)
                    ).mean()
                    train_attn_penalty += (
                        avg_attn_penalty.item() / args.gradient_accumulation_steps
                    )

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(
                        unet.parameters(), channel_mapper.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                    ema_rie.step(rie.parameters())
                    ema_channel_mapper.step(channel_mapper.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log(
                    {"train_attn_penalty": train_attn_penalty}, step=global_step
                )
                accelerator.log(
                    {"learning_rate": lr_scheduler.get_last_lr()[0]}, step=global_step
                )
                train_loss = 0.0

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step % args.show_steps == 0:
                accelerator.wait_for_everyone()
                # if accelerator.is_main_process:
                copy_unet = copy.deepcopy(accelerator.unwrap_model(unet))
                copy_channel_mapper = copy.deepcopy(
                    accelerator.unwrap_model(channel_mapper)
                )
                copy_rie = copy.deepcopy(accelerator.unwrap_model(rie))
                if args.use_ema:
                    ema_unet.copy_to(copy_unet.parameters())
                    ema_rie.copy_to(copy_rie.parameters())
                    ema_channel_mapper.copy_to(copy_channel_mapper.parameters())

                pipeline = PoseTransferPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    channel_mapper=copy_channel_mapper.to(dtype=weight_dtype).eval(),
                    rie=copy_rie.to(dtype=weight_dtype).eval(),
                    vae=vae,
                    unet=copy_unet.to(dtype=weight_dtype).eval(),
                )

                for batch in val_dataloader:
                    ref = batch["ref_values"].to(weight_dtype)
                    cat = batch["cat_values"].to(weight_dtype)
                    ground_truths = batch["pixel_values"]
                    constraint_maps = (
                        batch["constraint_maps"]
                        if args.use_constraint_penalty
                        else None
                    )
                    break

                images = pipeline(
                    ref=ref,
                    cat=cat,
                    height=args.height,
                    width=args.width,
                    generator=torch.Generator(device="cuda").manual_seed(0),
                    output_type="numpy",
                    ref_guidance_scale=2,
                    cat_guidance_scale=2,
                    constraint_maps=constraint_maps,
                ).images
                del pipeline
                del copy_unet
                del copy_channel_mapper
                del copy_rie
                torch.cuda.empty_cache()
                if accelerator.is_main_process:
                    images_processed = (images * 255).round().astype("uint8")
                    accelerator.trackers[0].writer.add_images(
                        "test_samples",
                        images_processed.transpose(0, 3, 1, 2),
                        global_step,
                    )
                    ground_truths = batch["pixel_values"].cpu().float()
                    ground_truths = (ground_truths / 2 + 0.5).clamp(0, 1) * 255
                    accelerator.trackers[0].writer.add_images(
                        "ground_truths",
                        ground_truths.numpy().round().astype("uint8"),
                        global_step,
                    )
                accelerator.wait_for_everyone()

            if global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    copy_unet = copy.deepcopy(accelerator.unwrap_model(unet))
                    copy_channel_mapper = copy.deepcopy(
                        accelerator.unwrap_model(channel_mapper)
                    )
                    copy_rie = copy.deepcopy(accelerator.unwrap_model(rie))
                    if args.use_ema:
                        ema_unet.copy_to(copy_unet.parameters())
                        ema_rie.copy_to(copy_rie.parameters())
                        ema_channel_mapper.copy_to(copy_channel_mapper.parameters())

                    pipeline = PoseTransferPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        channel_mapper=copy_channel_mapper,
                        rie=copy_rie,
                        vae=vae,
                        unet=copy_unet,
                    )
                    save_path = os.path.join(args.output_dir, f"step_{global_step}")
                    pipeline.save_pretrained(save_path)
                    logger.info(f"Saved pipeline to {save_path}")
                    del pipeline
                    del copy_unet
                    del copy_channel_mapper
                    del copy_rie
                    torch.cuda.empty_cache()

                    # For resume training only
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint_{global_step}"
                    )
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        channel_mapper = accelerator.unwrap_model(channel_mapper)
        rie = accelerator.unwrap_model(rie)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
            ema_rie.copy_to(rie.parameters())
            ema_channel_mapper.copy_to(channel_mapper.parameters())

        pipeline = PoseTransferPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            channel_mapper=channel_mapper,
            rie=rie,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(os.path.join(args.output_dir, "final"))

        if args.push_to_hub:
            repo.push_to_hub(
                commit_message="End of training", blocking=False, auto_lfs_prune=True
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
