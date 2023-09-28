import copy
from typing import Iterable, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import ResnetBlock2D, Downsample2D, Upsample2D
from diffusers.models.unet_2d_blocks import get_down_block
from diffusers.models.unet_2d_condition import (
    UNet2DConditionOutput,
    UNet2DConditionModel,
)
from diffusers.utils import logging
from perceiver_pytorch import Perceiver


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ChannelMapper(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 1,
        concat: bool = True,
        bias: bool = True,
        zero_init: bool = False,
    ):
        super().__init__()
        self.concat = concat
        self.cond_conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        if zero_init:
            self.cond_conv.weight.data.fill_(0.0)

    def forward(self, x, cond):
        cond = self.cond_conv(cond)
        if self.concat:
            return torch.cat([x, cond], dim=1)
        return x + cond


class ReferenceImageEncoder(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        down_block_types: Tuple[str] = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        block_out_channels: Tuple[int] = (128, 256, 512),
        multi_scale: bool = False,
        use_cls_token: bool = False,
    ):
        super().__init__()
        self.multi_scale = multi_scale
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)
        )
        self.down_blocks = nn.ModuleList([])
        if self.multi_scale:
            self.channel_mappers = nn.ModuleList([])
            for i in range(len(block_out_channels) - 1):
                channel_mapper = nn.Conv2d(
                    block_out_channels[i], block_out_channels[-1], kernel_size=1
                )
                self.channel_mappers.append(channel_mapper)

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            down_block = get_down_block(
                down_block_type,
                num_layers=1,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                add_downsample=True,
                resnet_eps=1e-5,
                resnet_act_fn="silu",
                attn_num_head_channels=8,
                resnet_groups=32,
                downsample_padding=1,
            )
            self.down_blocks.append(down_block)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, block_out_channels[-1]))
        else:
            self.cls_token = None

    @staticmethod
    def flatten(sample):
        batch, channel, height, width = sample.shape
        sample = sample.permute(0, 2, 3, 1).reshape(batch, height * width, channel)
        return sample

    def forward(self, sample):
        sample = self.conv_in(sample)
        if self.multi_scale:
            res_samples = []
            for i, downsample_block in enumerate(self.down_blocks):
                sample, _ = downsample_block(hidden_states=sample, temb=None)
                res_sample = sample
                if i != len(self.down_blocks) - 1:
                    res_sample = self.channel_mappers[i](res_sample)
                    res_sample = self.flatten(res_sample)
                else:
                    res_sample = self.flatten(res_sample)
                res_samples.append(res_sample)
            sample = torch.cat(res_samples, dim=1)
        else:
            for downsample_block in self.down_blocks:
                sample, _ = downsample_block(hidden_states=sample, temb=None)
            sample = self.flatten(sample)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(sample.shape[0], -1, -1)
            sample = torch.cat([cls_token, sample], dim=1)
        return sample


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.collected_params = None

        self.decay = decay
        self.optimization_step = 0

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        value = (1 + self.optimization_step) / (10 + self.optimization_step)
        one_minus_decay = 1 - min(self.decay, value)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.sub_(one_minus_decay * (s_param - param))
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype)
            if p.is_floating_point()
            else p.to(device=device)
            for p in self.shadow_params
        ]

    def state_dict(self) -> dict:
        r"""
        Returns the state of the ExponentialMovingAverage as a dict.
        This method is used by accelerate during checkpointing to save the ema state dict.
        """
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "optimization_step": self.optimization_step,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Loads the ExponentialMovingAverage state.
        This method is used by accelerate during checkpointing to save the ema state dict.
        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.optimization_step = state_dict["optimization_step"]
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.shadow_params = state_dict["shadow_params"]
        if not isinstance(self.shadow_params, list):
            raise ValueError("shadow_params must be a list")
        if not all(isinstance(p, torch.Tensor) for p in self.shadow_params):
            raise ValueError("shadow_params must all be Tensors")

        self.collected_params = state_dict["collected_params"]
        if self.collected_params is not None:
            if not isinstance(self.collected_params, list):
                raise ValueError("collected_params must be a list")
            if not all(isinstance(p, torch.Tensor) for p in self.collected_params):
                raise ValueError("collected_params must all be Tensors")
            if len(self.collected_params) != len(self.shadow_params):
                raise ValueError(
                    "collected_params and shadow_params must have the same length"
                )


class DoubleAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int,
        num_filters: int,
        heads: int = 8,
        dim_head: int = 64,
        bias=False,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
        ln_eps: float = 1e-5,
        use_constraint_penalty: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.norm = nn.LayerNorm(cross_attention_dim, eps=ln_eps)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.extraction_filters = nn.Parameter(torch.randn(num_filters, inner_dim))
        self.distribution_filters = nn.Parameter(torch.randn(num_filters, inner_dim))
        self.scale = nn.Parameter(torch.ones(1))

        self.use_constraint_penalty = use_constraint_penalty

    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.contiguous().reshape(
            batch_size // head_size, head_size, seq_len, dim
        )
        tensor = (
            tensor.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size // head_size, seq_len, dim * head_size)
        )
        return tensor

    def head_to_batch_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.contiguous().reshape(
            batch_size, seq_len, head_size, dim // head_size
        )
        tensor = (
            tensor.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size * head_size, seq_len, dim // head_size)
        )
        return tensor

    def get_attention_scores(self, query, key, dim=-1):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.bmm(
            query,
            key.transpose(-1, -2),
        )
        attention_scores = attention_scores * self.scale

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=dim)
        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def skip_forward(self, hidden_states):
        batch_size, _, height, width = hidden_states.shape
        hidden_states = (
            hidden_states.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(batch_size, -1, self.query_dim)
        )
        hidden_states = self.to_q(hidden_states)
        hidden_states = self.to_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch_size, height, width, self.query_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return hidden_states

    def forward(self, hidden_states, encoder_hidden_states, constraint_maps=None):
        residual = hidden_states
        batch_size, _, height, width = hidden_states.shape
        _, _, ref_height, ref_width = encoder_hidden_states.shape
        hidden_states = (
            hidden_states.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(batch_size, -1, self.query_dim)
        )
        encoder_hidden_states = (
            encoder_hidden_states.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(batch_size, -1, self.cross_attention_dim)
        )
        encoder_hidden_states = self.norm(encoder_hidden_states)

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        extraction_filters = self.extraction_filters.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        distribution_filters = self.distribution_filters.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        query = self.head_to_batch_dim(query)  # hw,c
        key = self.head_to_batch_dim(key)  # hw,c
        value = self.head_to_batch_dim(value)  # hw,c
        extraction_filters = self.head_to_batch_dim(extraction_filters)  # k,c
        distribution_filters = self.head_to_batch_dim(distribution_filters)  # k,c

        spatial_attention_probs = self.get_attention_scores(
            extraction_filters, key, dim=-1
        )  # k,hw
        channel_attention_probs = self.get_attention_scores(
            distribution_filters, query, dim=1
        )  # k,hw
        attention_probs = torch.bmm(
            channel_attention_probs.transpose(-1, -2), spatial_attention_probs
        )  # hw,hw

        attn_penalty = 0
        if constraint_maps is not None:
            if self.training and self.use_constraint_penalty:
                constraint_maps = constraint_maps.bool()
                constraint_maps = constraint_maps.reshape(
                    constraint_maps.shape[0], height * width, ref_height * ref_width
                )
                constraint_maps = (
                    constraint_maps.unsqueeze(1)
                    .expand(-1, self.heads, -1, -1)
                    .reshape(-1, height * width, ref_height * ref_width)
                    .permute(0, 2, 1)
                )
                filled_attention_probs = attention_probs.clone()
                filled_attention_probs[constraint_maps] = 1
                attn_penalty = F.mse_loss(
                    filled_attention_probs.float(),
                    constraint_maps.float(),
                    reduction="mean",
                )

        # attention_probs = self.get_attention_scores(query, key, dim=-1)
        hidden_states = torch.bmm(attention_probs, value)

        hidden_states = self.batch_to_head_dim(hidden_states)
        hidden_states = self.to_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch_size, height, width, self.query_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return hidden_states + residual * self.scale, attn_penalty


class DoubleAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_filters: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_heads=8,
        cross_attention_dim=512,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        upcast_attention=True,
        upcast_softmax=True,
        use_constraint_penalty: bool = False,
        skip_if_none: bool = True,
    ):
        super().__init__()
        resnets = []
        attentions = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                DoubleAttention(
                    query_dim=out_channels,
                    cross_attention_dim=cross_attention_dim,
                    num_filters=num_filters,
                    heads=attn_num_heads,
                    dim_head=out_channels // attn_num_heads,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                    ln_eps=resnet_eps,
                    use_constraint_penalty=use_constraint_penalty,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False
        self.skip_if_none = skip_if_none

    def forward(
        self, hidden_states, temb=None, encoder_hidden_states=None, constraint_maps=None
    ):
        output_states = ()
        attn_penalty = 0

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            if encoder_hidden_states is not None:
                hidden_states, p = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    constraint_maps=constraint_maps,
                )
                attn_penalty = attn_penalty + p
            else:
                if not self.skip_if_none:
                    hidden_states = attn.skip_forward(hidden_states)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states, attn_penalty


class DoubleAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        num_filters: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_heads=8,
        cross_attention_dim=512,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_upsample=True,
        upcast_attention=True,
        upcast_softmax=True,
        use_constraint_penalty: bool = False,
        skip_if_none: bool = True,
    ):
        super().__init__()
        resnets = []
        attentions = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                DoubleAttention(
                    query_dim=out_channels,
                    cross_attention_dim=cross_attention_dim,
                    num_filters=num_filters,
                    heads=attn_num_heads,
                    dim_head=out_channels // attn_num_heads,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                    ln_eps=resnet_eps,
                    use_constraint_penalty=use_constraint_penalty,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.skip_if_none = skip_if_none

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
        constraint_maps=None,
    ):
        attn_penalty = 0

        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            if encoder_hidden_states is not None:
                hidden_states, p = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    constraint_maps=constraint_maps,
                )
                attn_penalty = attn_penalty + p
            else:
                if not self.skip_if_none:
                    hidden_states = attn.skip_forward(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states, attn_penalty


class UNetMidBlock2DDoubleAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        num_filters: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_heads=8,
        cross_attention_dim=512,
        output_scale_factor=1.0,
        upcast_attention=True,
        upcast_softmax=True,
        use_constraint_penalty: bool = False,
        skip_if_none: bool = True,
    ):
        super().__init__()

        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            attentions.append(
                DoubleAttention(
                    query_dim=in_channels,
                    cross_attention_dim=cross_attention_dim,
                    num_filters=num_filters,
                    heads=attn_num_heads,
                    dim_head=in_channels // attn_num_heads,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                    ln_eps=resnet_eps,
                    use_constraint_penalty=use_constraint_penalty,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.skip_if_none = skip_if_none

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        constraint_maps=None,
    ):
        attn_penalty = 0

        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if encoder_hidden_states is not None:
                hidden_states, p = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    constraint_maps=constraint_maps,
                )
                attn_penalty = attn_penalty + p
            else:
                if not self.skip_if_none:
                    hidden_states = attn.skip_forward(hidden_states)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states, attn_penalty


class UNet2DDoubleAttentionConditionModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        num_filters: Union[int, Tuple[int]] = 64,
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_channels: Tuple[int] = (128, 256, 512, 512),
        attn_heads_nums: Union[int, Tuple[int]] = 8,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
        resnet_time_scale_shift: str = "default",
        use_constraint_penalty: bool = False,
        skip_if_none: bool = True,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)
        )

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(attn_heads_nums, int):
            attn_heads_nums = (attn_heads_nums,) * len(block_out_channels)
        if isinstance(num_filters, int):
            num_filters = (num_filters,) * len(block_out_channels)

        # down
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DoubleAttnDownBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_filters=num_filters[i],
                num_layers=layers_per_block,
                resnet_eps=norm_eps,
                resnet_time_scale_shift=resnet_time_scale_shift,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_heads=attn_heads_nums[i],
                cross_attention_dim=cross_attention_channels[i],
                add_downsample=not is_final_block,
                upcast_attention=upcast_attention,
                upcast_softmax=upcast_softmax,
                use_constraint_penalty=use_constraint_penalty,
                skip_if_none=skip_if_none,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2DDoubleAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            num_filters=num_filters[-1],
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_groups=norm_num_groups,
            attn_num_heads=attn_heads_nums[-1],
            cross_attention_dim=cross_attention_channels[-1],
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            use_constraint_penalty=use_constraint_penalty,
            skip_if_none=skip_if_none,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attn_heads_nums = list(reversed(attn_heads_nums))
        reversed_cross_attention_channels = list(reversed(cross_attention_channels))
        reversed_num_filters = list(reversed(num_filters))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = DoubleAttnUpBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                num_filters=reversed_num_filters[i],
                add_upsample=add_upsample,
                num_layers=layers_per_block + 1,
                resnet_eps=norm_eps,
                resnet_time_scale_shift=resnet_time_scale_shift,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_heads=reversed_attn_heads_nums[i],
                cross_attention_dim=reversed_cross_attention_channels[i],
                upcast_attention=upcast_attention,
                upcast_softmax=upcast_softmax,
                use_constraint_penalty=use_constraint_penalty,
                skip_if_none=skip_if_none,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    @staticmethod
    @torch.no_grad()
    def get_constraint_maps(constraint_maps, scale_factor):
        to_maps, from_maps = constraint_maps.chunk(2, dim=1)
        to_maps = F.interpolate(to_maps, scale_factor=scale_factor, mode="nearest")
        from_maps = F.interpolate(from_maps, scale_factor=scale_factor, mode="nearest")
        batch_size = to_maps.shape[0]
        constraint_maps = (
            to_maps.reshape(batch_size, 1, -1)
            .eq(from_maps.reshape(batch_size, -1, 1))
            .float()
        )
        return constraint_maps

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        constraint_maps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:

        default_overall_up_factor = 2 ** self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        constraint_maps_list = []
        for i in range(len(self.down_blocks)):
            if constraint_maps is None:
                constraint_maps_list.append(None)
            else:
                constraint_maps_list.append(
                    self.get_constraint_maps(constraint_maps, 1 / 2 ** i)
                )

        # 3. down
        down_block_res_samples = (sample,)
        attn_penalty = 0
        if encoder_hidden_states is None:
            encoder_hidden_states = [None] * len(self.down_blocks)
        for i, downsample_block in enumerate(self.down_blocks):
            sample, res_samples, p = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states[i],
                constraint_maps=constraint_maps_list[i],
            )
            down_block_res_samples += res_samples
            attn_penalty = attn_penalty + p

        # 4. mid
        sample, p = self.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states[-1],
            constraint_maps=constraint_maps_list[-1],
        )
        attn_penalty = attn_penalty + p

        reversed_encoder_hidden_states = encoder_hidden_states[::-1]
        reversed_constraint_maps_list = constraint_maps_list[::-1]

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            sample, p = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=reversed_encoder_hidden_states[i],
                upsample_size=upsample_size,
                constraint_maps=reversed_constraint_maps_list[i],
            )
            attn_penalty = attn_penalty + p

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        if self.training:
            return UNet2DConditionOutput(sample=sample), attn_penalty
        return UNet2DConditionOutput(sample=sample)


class MultiScaleReferenceImageEncoder(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        down_block_types: Tuple[str] = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        num_layers: int = 1,
        norm_eps: float = 1e-5,
        use_time_emb: Optional[bool] = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)
        )
        self.down_blocks = nn.ModuleList([])

        # time
        if use_time_emb:
            timestep_input_dim = block_out_channels[0]
            time_embed_dim = block_out_channels[0] * 4
            self.time_proj = Timesteps(
                block_out_channels[0], flip_sin_to_cos, freq_shift
            )
            self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        else:
            time_embed_dim = None
            self.time_proj = None
            self.time_embedding = None

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            down_block = get_down_block(
                down_block_type,
                num_layers=num_layers,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=i > 0,
                resnet_eps=norm_eps,
                resnet_act_fn="silu",
                attn_num_head_channels=1,
                resnet_groups=32,
                downsample_padding=1,
            )
            self.down_blocks.append(down_block)

    def forward(self, sample, timesteps=None):
        sample = self.conv_in(sample)
        samples = []
        if timesteps is not None and self.config.use_time_emb:
            # timesteps = timestep
            timesteps = timesteps.expand(sample.shape[0])
            temb = self.time_proj(timesteps)
            temb = temb.to(dtype=self.dtype)
            temb = self.time_embedding(temb)
        else:
            temb = None
        for i, downsample_block in enumerate(self.down_blocks):
            sample, _ = downsample_block(hidden_states=sample, temb=temb)
            samples.append(sample)

        return samples
