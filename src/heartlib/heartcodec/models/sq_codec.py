"""Scalar-quantised convolutional audio codec (SQ codec).

This module provides :class:`ScalarModel`, a convolutional encoder/decoder
that operates on raw audio waveforms with scalar quantisation, along with
all supporting building blocks: causal/non-causal convolutions, Snake
activations, residual units, and up/downsample layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm
from torch.autograd.function import InplaceFunction


def get_padding(kernel_size, dilation=1):
    """Compute symmetric padding for a convolution to preserve sequence length."""
    return int((kernel_size * dilation - dilation) / 2)


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    """Snake activation: ``x + (1/alpha) * sin(alpha * x)^2``."""
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """1-D Snake activation with a learnable per-channel frequency parameter."""

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class Conv1d(nn.Conv1d):
    """1-D convolution with optional causal left-padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
        padding=None,
        causal: bool = False,
        w_init_gain=None,
    ):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = get_padding(kernel_size, dilation)
        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias,
        )
        if w_init_gain is not None:
            torch.nn.init.xavier_uniform_(
                self.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
            )

    def forward(self, x):
        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        return super(Conv1d, self).forward(x)


class ConvTranspose1d(nn.ConvTranspose1d):
    """1-D transposed convolution with optional causal trimming."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding=None,
        padding_mode: str = "zeros",
        causal: bool = False,
    ):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert padding == 0, "padding is not allowed in causal ConvTranspose1d."
            assert (
                kernel_size == 2 * stride
            ), "kernel_size must be equal to 2*stride is not allowed in causal ConvTranspose1d."
        super(ConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.causal = causal
        self.stride = stride

    def forward(self, x):
        x = super(ConvTranspose1d, self).forward(x)
        if self.causal:
            x = x[:, :, : -self.stride]
        return x


class PreProcessor(nn.Module):
    """Convolve then average-pool to downsample by *num_samples*."""

    def __init__(self, n_in, n_out, num_samples, kernel_size=7, causal=False):
        super(PreProcessor, self).__init__()
        self.pooling = torch.nn.AvgPool1d(kernel_size=num_samples)
        self.conv = Conv1d(n_in, n_out, kernel_size=kernel_size, causal=causal)
        self.activation = nn.PReLU()

    def forward(self, x):
        output = self.activation(self.conv(x))
        output = self.pooling(output)
        return output


class PostProcessor(nn.Module):
    """Repeat-upsample by *num_samples* then convolve."""

    def __init__(self, n_in, n_out, num_samples, kernel_size=7, causal=False):
        super(PostProcessor, self).__init__()
        self.num_samples = num_samples
        self.conv = Conv1d(n_in, n_out, kernel_size=kernel_size, causal=causal)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        B, T, C = x.size()
        x = x.repeat(1, 1, self.num_samples).view(B, -1, C)
        x = torch.transpose(x, 1, 2)
        output = self.activation(self.conv(x))
        return output


class ResidualUnit(nn.Module):
    """Dilated residual convolution block with two weight-normed 1-D convolutions."""

    def __init__(self, n_in, n_out, dilation, res_kernel_size=7, causal=False):
        super(ResidualUnit, self).__init__()
        self.conv1 = weight_norm(
            Conv1d(
                n_in,
                n_out,
                kernel_size=res_kernel_size,
                dilation=dilation,
                causal=causal,
            )
        )
        self.conv2 = weight_norm(Conv1d(n_in, n_out, kernel_size=1, causal=causal))
        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()

    def forward(self, x):
        output = self.activation1(self.conv1(x))
        output = self.activation2(self.conv2(output))
        return output + x


class ResEncoderBlock(nn.Module):
    """Encoder block: a chain of dilated residual units followed by a downsample layer."""

    def __init__(
        self, n_in, n_out, stride, down_kernel_size, res_kernel_size=7, causal=False
    ):
        super(ResEncoderBlock, self).__init__()
        self.convs = nn.ModuleList(
            [
                ResidualUnit(
                    n_in,
                    n_out // 2,
                    dilation=1,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out // 2,
                    n_out // 2,
                    dilation=3,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out // 2,
                    n_out // 2,
                    dilation=5,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out // 2,
                    n_out // 2,
                    dilation=7,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out // 2,
                    n_out // 2,
                    dilation=9,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
            ]
        )

        self.down_conv = DownsampleLayer(
            n_in, n_out, down_kernel_size, stride=stride, causal=causal
        )

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.down_conv(x)
        return x


class ResDecoderBlock(nn.Module):
    """Decoder block: an upsample layer followed by a chain of dilated residual units."""

    def __init__(
        self, n_in, n_out, stride, up_kernel_size, res_kernel_size=7, causal=False
    ):
        super(ResDecoderBlock, self).__init__()
        self.up_conv = UpsampleLayer(
            n_in,
            n_out,
            kernel_size=up_kernel_size,
            stride=stride,
            causal=causal,
            activation=None,
        )

        self.convs = nn.ModuleList(
            [
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=1,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=3,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=5,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=7,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=9,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
            ]
        )

    def forward(self, x):
        x = self.up_conv(x)
        for conv in self.convs:
            x = conv(x)
        return x


class DownsampleLayer(nn.Module):
    """Strided or pooled convolution layer for temporal downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        activation=nn.PReLU(),
        use_weight_norm: bool = True,
        pooling: bool = False,
    ):
        super(DownsampleLayer, self).__init__()
        self.pooling = pooling
        self.stride = stride
        self.activation = nn.PReLU()
        self.use_weight_norm = use_weight_norm
        if pooling:
            self.layer = Conv1d(in_channels, out_channels, kernel_size, causal=causal)
            self.pooling = nn.AvgPool1d(kernel_size=stride)
        else:
            self.layer = Conv1d(
                in_channels, out_channels, kernel_size, stride=stride, causal=causal
            )
        if use_weight_norm:
            self.layer = weight_norm(self.layer)

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x) if self.activation is not None else x
        if self.pooling:
            x = self.pooling(x)
        return x

    def remove_weight_norm(self):
        if self.use_weight_norm:
            remove_weight_norm(self.layer)


class UpsampleLayer(nn.Module):
    """Transposed-convolution or repeat-based temporal upsampling layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        activation=nn.PReLU(),
        use_weight_norm: bool = True,
        repeat: bool = False,
    ):
        super(UpsampleLayer, self).__init__()
        self.repeat = repeat
        self.stride = stride
        self.activation = activation
        self.use_weight_norm = use_weight_norm
        if repeat:
            self.layer = Conv1d(in_channels, out_channels, kernel_size, causal=causal)
        else:
            self.layer = ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride=stride, causal=causal
            )
        if use_weight_norm:
            self.layer = weight_norm(self.layer)

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x) if self.activation is not None else x
        if self.repeat:
            x = torch.transpose(x, 1, 2)
            B, T, C = x.size()
            x = x.repeat(1, 1, self.stride).view(B, -1, C)
            x = torch.transpose(x, 1, 2)
        return x

    def remove_weight_norm(self):
        if self.use_weight_norm:
            remove_weight_norm(self.layer)


class round_func9(InplaceFunction):
    """Scalar quantiser that rounds to 9 uniformly spaced levels with straight-through gradients."""

    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(9 * input) / 9

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class ScalarModel(nn.Module):
    """Scalar-quantised convolutional encoder/decoder for audio.

    The encoder compresses a waveform to a compact latent representation
    via stacked residual encoder blocks, applies a ``tanh`` activation, and
    scalar-quantises the result to 9 levels.  The decoder mirrors the
    encoder with upsampling blocks to reconstruct the waveform.

    Args:
        num_bands: Number of input frequency bands (1 for mono).
        sample_rate: Audio sample rate in Hz.
        causal: Whether to use causal convolutions throughout.
        num_samples: Pre/post-processor resampling factor.
        downsample_factors: Per-stage downsampling strides for the encoder.
        downsample_kernel_sizes: Kernel sizes for each encoder stage.
        upsample_factors: Per-stage upsampling strides for the decoder.
        upsample_kernel_sizes: Kernel sizes for each decoder stage.
        latent_hidden_dim: Channel count of the bottleneck latent.
        default_kernel_size: Kernel size for pre/post-processor convolutions.
        delay_kernel_size: Kernel size for the decoder look-ahead
            convolution.
        init_channel: Base channel count that doubles with each stage.
        res_kernel_size: Kernel size used in residual units.
        mode: Unused legacy parameter (kept for checkpoint compatibility).
    """

    def __init__(
        self,
        num_bands,
        sample_rate,
        causal,
        num_samples,
        downsample_factors,
        downsample_kernel_sizes,
        upsample_factors,
        upsample_kernel_sizes,
        latent_hidden_dim,
        default_kernel_size,
        delay_kernel_size,
        init_channel,
        res_kernel_size,
        mode="pre_proj",
    ):
        super(ScalarModel, self).__init__()
        # self.args = args
        self.encoder = []
        self.decoder = []
        self.vq = round_func9()  # using 9
        self.mode = mode
        # Encoder parts
        self.encoder.append(
            weight_norm(
                Conv1d(
                    num_bands,
                    init_channel,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        )
        if num_samples > 1:
            # Downsampling
            self.encoder.append(
                PreProcessor(
                    init_channel,
                    init_channel,
                    num_samples,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        for i, down_factor in enumerate(downsample_factors):
            self.encoder.append(
                ResEncoderBlock(
                    init_channel * np.power(2, i),
                    init_channel * np.power(2, i + 1),
                    down_factor,
                    downsample_kernel_sizes[i],
                    res_kernel_size,
                    causal=causal,
                )
            )
        self.encoder.append(
            weight_norm(
                Conv1d(
                    init_channel * np.power(2, len(downsample_factors)),
                    latent_hidden_dim,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        )
        # Decoder
        # look ahead
        self.decoder.append(
            weight_norm(
                Conv1d(
                    latent_hidden_dim,
                    init_channel * np.power(2, len(upsample_factors)),
                    kernel_size=delay_kernel_size,
                )
            )
        )
        for i, upsample_factor in enumerate(upsample_factors):
            self.decoder.append(
                ResDecoderBlock(
                    init_channel * np.power(2, len(upsample_factors) - i),
                    init_channel * np.power(2, len(upsample_factors) - i - 1),
                    upsample_factor,
                    upsample_kernel_sizes[i],
                    res_kernel_size,
                    causal=causal,
                )
            )
        if num_samples > 1:
            self.decoder.append(
                PostProcessor(
                    init_channel,
                    init_channel,
                    num_samples,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        self.decoder.append(
            weight_norm(
                Conv1d(
                    init_channel,
                    num_bands,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        )
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)

    def forward(self, x):
        """Encode, quantise, and decode the input waveform *x*."""
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x))
        # import pdb; pdb.set_trace()
        x = self.vq.apply(x)  # vq
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return x

    def inference(self, x):
        """Run encode → quantise → decode and return all intermediate tensors.

        Returns:
            A tuple of ``(embedding, quantised_embedding, reconstruction)``.
        """
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x))  # reverse to tanh

        emb = x
        # import pdb; pdb.set_trace()
        emb_quant = self.vq.apply(emb)  # vq
        x = emb_quant
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return emb, emb_quant, x

    def encode(self, x):
        """Encode the waveform *x* into a continuous (pre-quantised) latent."""
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x))  # reverse to tanh

        emb = x
        # import pdb; pdb.set_trace()
        emb_quant = self.vq.apply(emb)  # vq
        return emb

    def decode(self, x):
        """Quantise then decode a latent tensor *x* to reconstruct the waveform."""
        x = self.vq.apply(
            x
        )  # make sure the prediction follow the similar disctribution
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return x
