"""Configuration for the HeartCodec audio codec model."""

from transformers.configuration_utils import PretrainedConfig
from typing import List


class HeartCodecConfig(PretrainedConfig):
    """Configuration class for :class:`HeartCodec`.

    HeartCodec is a 12.5 Hz music codec composed of three sub-modules: a
    residual vector quantiser (RVQ), a flow-matching diffusion transformer,
    and a scalar convolutional encoder/decoder (SQ codec).  This config
    stores the hyper-parameters for all three.

    Args:
        dim: RVQ embedding dimension.
        codebook_size: Number of entries per RVQ codebook.
        decay: EMA decay factor for codebook updates.
        commitment_weight: Weight of the commitment loss term.
        threshold_ema_dead_code: Minimum usage count before a code is
            considered dead and eligible for re-initialisation.
        use_cosine_sim: Whether to use cosine similarity for codebook
            look-ups instead of L2 distance.
        codebook_dim: Dimensionality of each codebook entry (may differ
            from ``dim`` when a projection is used).
        num_quantizers: Number of residual quantisation stages.
        attention_head_dim: Dimension of each attention head in the
            diffusion transformer.
        in_channels: Input channel count for the diffusion transformer.
        norm_type: Normalisation type used by the transformer
            (e.g. ``"ada_norm_single"``).
        num_attention_heads: Number of attention heads in the diffusion
            transformer.
        num_layers: Number of transformer blocks in the first stage.
        num_layers_2: Number of transformer blocks in the second stage.
        out_channels: Output channel count of the diffusion transformer.
        num_bands: Number of frequency bands for the SQ codec input.
        sample_rate: Audio sample rate in Hz.
        causal: Whether to use causal convolutions in the SQ codec.
        num_samples: Pre/post-processor resampling factor.
        downsample_factors: Per-stage downsampling factors for the encoder.
        downsample_kernel_sizes: Kernel sizes for each downsample stage.
        upsample_factors: Per-stage upsampling factors for the decoder.
        upsample_kernel_sizes: Kernel sizes for each upsample stage.
        latent_hidden_dim: Hidden dimension of the SQ codec latent space.
        default_kernel_size: Default kernel size for pre/post-processor
            convolutions.
        delay_kernel_size: Kernel size for the decoder look-ahead
            convolution.
        init_channel: Base channel count for the SQ codec encoder/decoder.
        res_kernel_size: Kernel size used in residual units.
    """

    model_type = "heartcodec"

    def __init__(
        self,
        # config for rvq
        dim: int = 512,
        codebook_size: int = 8192,
        decay: float = 0.9,
        commitment_weight: float = 1.0,
        threshold_ema_dead_code: int = 2,
        use_cosine_sim: bool = False,
        codebook_dim: int = 32,
        num_quantizers: int = 8,
        # config for diffusion transformer
        attention_head_dim: int = 64,
        in_channels: int = 1024,
        norm_type: str = "ada_norm_single",
        num_attention_heads: int = 24,
        num_layers: int = 24,
        num_layers_2: int = 6,
        out_channels: int = 256,
        # config for sq codec
        num_bands: int = 1,
        sample_rate: int = 48000,
        causal: bool = True,
        num_samples: int = 2,
        downsample_factors: List[int] = [3, 4, 4, 4, 5],
        downsample_kernel_sizes: List[int] = [6, 8, 8, 8, 10],
        upsample_factors: List[int] = [5, 4, 4, 4, 3],
        upsample_kernel_sizes: List[int] = [10, 8, 8, 8, 6],
        latent_hidden_dim: int = 128,
        default_kernel_size: int = 7,
        delay_kernel_size: int = 5,
        init_channel: int = 64,
        res_kernel_size: int = 7,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.use_cosine_sim = use_cosine_sim
        self.codebook_dim = codebook_dim
        self.num_quantizers = num_quantizers

        self.attention_head_dim = attention_head_dim
        self.in_channels = in_channels
        self.norm_type = norm_type
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.num_layers_2 = num_layers_2
        self.out_channels = out_channels

        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.causal = causal
        self.num_samples = num_samples
        self.downsample_factors = downsample_factors
        self.downsample_kernel_sizes = downsample_kernel_sizes
        self.upsample_factors = upsample_factors
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.latent_hidden_dim = latent_hidden_dim
        self.default_kernel_size = default_kernel_size
        self.delay_kernel_size = delay_kernel_size
        self.init_channel = init_channel
        self.res_kernel_size = res_kernel_size
