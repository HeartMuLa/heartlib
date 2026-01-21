import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from vector_quantize_pytorch import ResidualVQ
from .transformer import LlamaTransformer


class FlowMatching(nn.Module):
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
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
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
