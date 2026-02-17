"""Flow-matching diffusion module for the HeartCodec pipeline.

This module implements :class:`FlowMatching`, which learns to map discrete
quantised audio codes to continuous latent representations using an
ODE-based flow-matching formulation solved with a fixed-step Euler
integrator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from vector_quantize_pytorch import ResidualVQ
from .transformer import LlamaTransformer


class FlowMatching(nn.Module):
    """Flow-matching model that converts discrete codes to continuous latents.

    The module contains a residual vector quantiser (RVQ) whose codebook
    embeddings serve as conditioning, and a :class:`LlamaTransformer`
    network that acts as the velocity estimator for the flow ODE.

    Args:
        dim: RVQ embedding dimension.
        codebook_size: Number of entries per RVQ codebook.
        decay: EMA decay for codebook updates.
        commitment_weight: Weight of the commitment loss.
        threshold_ema_dead_code: Minimum usage before a code is
            re-initialised.
        use_cosine_sim: Use cosine similarity for codebook look-ups.
        codebook_dim: Per-entry codebook dimensionality.
        num_quantizers: Number of residual quantisation stages.
        attention_head_dim: Head dimension for the transformer estimator.
        in_channels: Input channels of the transformer.
        norm_type: Normalisation type for the transformer.
        num_attention_heads: Number of attention heads.
        num_layers: Transformer blocks in the first stage.
        num_layers_2: Transformer blocks in the second stage.
        out_channels: Output channels of the transformer (latent dim).
    """

    def __init__(
        self,
        # rvq stuff
        dim: int = 512,
        codebook_size: int = 8192,
        decay: float = 0.9,
        commitment_weight: float = 1.0,
        threshold_ema_dead_code: int = 2,
        use_cosine_sim: bool = False,
        codebook_dim: int = 32,
        num_quantizers: int = 8,
        # dit backbone stuff
        attention_head_dim: int = 64,
        in_channels: int = 1024,
        norm_type: str = "ada_norm_single",
        num_attention_heads: int = 24,
        num_layers: int = 24,
        num_layers_2: int = 6,
        out_channels: int = 256,
    ):
        super().__init__()

        self.vq_embed = ResidualVQ(
            dim=dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=commitment_weight,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_cosine_sim=use_cosine_sim,
            codebook_dim=codebook_dim,
            num_quantizers=num_quantizers,
        )
        self.cond_feature_emb = nn.Linear(dim, dim)
        self.zero_cond_embedding1 = nn.Parameter(torch.randn(dim))
        self.estimator = LlamaTransformer(
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            norm_type=norm_type,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            num_layers_2=num_layers_2,
            out_channels=out_channels,
        )

        self.latent_dim = out_channels

    @torch.no_grad()
    def inference_codes(
        self,
        codes,
        true_latents,
        latent_length,
        incontext_length,
        guidance_scale=2.0,
        num_steps=20,
        disable_progress=True,
        scenario="start_seg",
    ):
        """Decode discrete codes into continuous latent representations.

        Quantised code embeddings are looked up via the RVQ, interpolated
        to the target temporal resolution, and used to condition the flow-
        matching ODE that is solved with :meth:`solve_euler`.

        Args:
            codes: List containing one integer code tensor of shape
                ``(B, num_quantizers, T)``.
            true_latents: Latent tensor seeded with random noise (and
                optionally in-context latents from a previous segment).
            latent_length: Number of latent frames to generate.
            incontext_length: Number of leading latent frames to treat as
                in-context (carried over from the previous segment).
            guidance_scale: Classifier-free guidance scale.
            num_steps: Number of Euler integration steps.
            disable_progress: Suppress the tqdm progress bar.
            scenario: Either ``"start_seg"`` for the first segment or
                ``"other_seg"`` for subsequent overlapping segments.

        Returns:
            A latent tensor of shape ``(B, T, latent_dim)``.
        """
        device = true_latents.device
        dtype = true_latents.dtype
        # codes_bestrq_middle, codes_bestrq_last = codes
        codes_bestrq_emb = codes[0]

        batch_size = codes_bestrq_emb.shape[0]
        self.vq_embed.eval()
        quantized_feature_emb = self.vq_embed.get_output_from_indices(
            codes_bestrq_emb.transpose(1, 2)
        )
        quantized_feature_emb = self.cond_feature_emb(quantized_feature_emb)  # b t 512
        # assert 1==2
        quantized_feature_emb = F.interpolate(
            quantized_feature_emb.permute(0, 2, 1), scale_factor=2, mode="nearest"
        ).permute(0, 2, 1)

        num_frames = quantized_feature_emb.shape[1]  #
        latents = torch.randn(
            (batch_size, num_frames, self.latent_dim), device=device, dtype=dtype
        )
        latent_masks = torch.zeros(
            latents.shape[0], latents.shape[1], dtype=torch.int64, device=latents.device
        )
        latent_masks[:, 0:latent_length] = 2
        if scenario == "other_seg":
            latent_masks[:, 0:incontext_length] = 1

        quantized_feature_emb = (latent_masks > 0.5).unsqueeze(
            -1
        ) * quantized_feature_emb + (latent_masks < 0.5).unsqueeze(
            -1
        ) * self.zero_cond_embedding1.unsqueeze(
            0
        )

        incontext_latents = (
            true_latents
            * ((latent_masks > 0.5) * (latent_masks < 1.5)).unsqueeze(-1).float()
        )
        incontext_length = ((latent_masks > 0.5) * (latent_masks < 1.5)).sum(-1)[0]

        additional_model_input = torch.cat([quantized_feature_emb], 1)
        temperature = 1.0
        t_span = torch.linspace(
            0, 1, num_steps + 1, device=quantized_feature_emb.device
        )
        latents = self.solve_euler(
            latents * temperature,
            incontext_latents.to(dtype),
            incontext_length,
            t_span,
            additional_model_input,
            guidance_scale,
        )

        latents[:, 0:incontext_length, :] = incontext_latents[
            :, 0:incontext_length, :
        ]  # B, T, dim
        return latents

    def solve_euler(self, x, incontext_x, incontext_length, t_span, mu, guidance_scale):
        """Solve the flow ODE with a fixed-step Euler integrator.

        At each step the in-context portion of *x* is interpolated between
        the initial noise and the true in-context latents, then the velocity
        field is evaluated (with optional classifier-free guidance) and the
        state is advanced.

        Args:
            x: Initial noise tensor of shape ``(B, T, D)``.
            incontext_x: Ground-truth latents for the in-context region.
            incontext_length: Number of leading frames that are in-context.
            t_span: Time grid of shape ``(num_steps + 1,)`` from 0 to 1.
            mu: Conditioning signal (quantised code embeddings).
            guidance_scale: Classifier-free guidance scale.

        Returns:
            The denoised latent tensor at ``t = 1``.
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        noise = x.clone()
        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        for step in tqdm(range(1, len(t_span))):
            x[:, 0:incontext_length, :] = (1 - (1 - 1e-6) * t) * noise[
                :, 0:incontext_length, :
            ] + t * incontext_x[:, 0:incontext_length, :]
            if guidance_scale > 1.0:
                dphi_dt = self.estimator(
                    torch.cat(
                        [
                            torch.cat([x, x], 0),
                            torch.cat([incontext_x, incontext_x], 0),
                            torch.cat([torch.zeros_like(mu), mu], 0),
                        ],
                        2,
                    ),
                    timestep=t.unsqueeze(-1).repeat(2),
                )
                dphi_dt_uncond, dhpi_dt_cond = dphi_dt.chunk(2, 0)
                dphi_dt = dphi_dt_uncond + guidance_scale * (
                    dhpi_dt_cond - dphi_dt_uncond
                )
            else:
                dphi_dt = self.estimator(
                    torch.cat([x, incontext_x, mu], 2), timestep=t.unsqueeze(-1)
                )

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        result = sol[-1]

        return result
