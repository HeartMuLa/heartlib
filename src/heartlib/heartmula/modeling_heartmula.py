import torch
import torch.nn as nn
import torch.nn.functional as F
from .configuration_heartmula import HeartMuLaConfig
from transformers.modeling_utils import PreTrainedModel
import torch
import torch.nn as nn
import torchtune
from torchtune.models import llama3_2


class FP8Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
            dtype=None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if not hasattr(torch, 'float8_e4m3fn'):
            raise ImportError("PyTorch 2.1+ required for Float8 support.")

        factory_kwargs = {'device': device, 'dtype': dtype or torch.bfloat16}
        init_weight = torch.empty((out_features, in_features), **factory_kwargs)
        nn.init.kaiming_uniform_(init_weight, a=5 ** 0.5)

        self.weight = nn.Parameter(init_weight.to(torch.float8_e4m3fn), requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(init_weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        comp_dtype = input.dtype
        weight_bf16 = self.weight.to(comp_dtype)
        return F.linear(input, weight_bf16, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def reset_parameters(self):
        # Helper to re-init if needed
        # We must init in high precision then cast down
        temp_weight = torch.empty((self.out_features, self.in_features), dtype=torch.bfloat16)
        nn.init.kaiming_uniform_(temp_weight, a=5 ** 0.5)
        with torch.no_grad():
            self.weight.copy_(temp_weight.to(torch.float8_e4m3fn))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(temp_weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


    @property
    def shape(self):
        """Allows accessing .shape like a tensor"""
        return self.weight.shape


def replace_linear_with_fp8(model):
    """
    Recursively replaces nn.Linear with FP8Linear.
    Transfers existing weights (if any) to the new layers.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_layer = FP8Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=(module.bias is not None),
                device=module.weight.device
            )
            with torch.no_grad():
                new_layer.weight.data.copy_(module.weight.data.to(torch.float8_e4m3fn))
                if module.bias is not None:
                    new_layer.bias.data.copy_(module.bias.data)

            setattr(model, name, new_layer)
        else:
            replace_linear_with_fp8(module)


def llama3_2_3B() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=28,
        num_heads=24,
        num_kv_heads=8,
        embed_dim=3072,
        max_seq_len=8192,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_300M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=3,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=3072,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_7B() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=8192,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_400M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=3072,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )  # 减少了num_heads和num_kv_heads之间的倍速，提升了精确度，但降低了效率


FLAVORS = {
    "llama-3B": llama3_2_3B,
    "llama-300M": llama3_2_300M,
    "llama-7B": llama3_2_7B,
    "llama-400M": llama3_2_400M,
}


def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()

    replace_linear_with_fp8(model)

    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    r = mask[input_pos, :]
    return r


def _multinomial_sample_one_no_sync(
    probs,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


class HeartMuLa(PreTrainedModel):
    config_class = HeartMuLaConfig

    def __init__(
        self,
        config: HeartMuLaConfig,
    ):
        super(HeartMuLa, self).__init__(config)

        self.config = config

        self.backbone, backbone_dim = _prepare_transformer(
            FLAVORS[config.backbone_flavor]()
        )
        self.decoder, decoder_dim = _prepare_transformer(
            FLAVORS[config.decoder_flavor]()
        )

        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.audio_num_codebooks, backbone_dim
        )
        self.unconditional_text_embedding = nn.Embedding(1, backbone_dim)

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(
            backbone_dim, config.audio_vocab_size, bias=False
        )
        self.audio_head = nn.Parameter(
            torch.empty(
                config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size
            )
        )
        self.muq_linear = nn.Linear(config.muq_dim, backbone_dim)
        self.post_init()

    def setup_caches(self, max_batch_size: int):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        try:
            self.reset_caches()
        except RuntimeError:
            pass

        with device:
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(
                max_batch_size,
                dtype,
                decoder_max_seq_len=self.config.audio_num_codebooks,
            )

        self.register_buffer(
            "backbone_causal_mask",
            _create_causal_mask(self.backbone.max_seq_len, device),
        )
        self.register_buffer(
            "decoder_causal_mask",
            _create_causal_mask(self.config.audio_num_codebooks, device),
        )

    @torch.inference_mode()
    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
        cfg_scale: float,
        continuous_segments: torch.Tensor = None,
        starts=None,
    ) -> torch.Tensor:
        b, s, _ = tokens.size()

        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)

        uncond_mask = None
        if cfg_scale > 1.0 and b > 1:
            actual_B = b // 2
            uncond_mask = torch.cat(
                [
                    torch.zeros(actual_B, dtype=torch.bool, device=tokens.device),
                    torch.ones(actual_B, dtype=torch.bool, device=tokens.device),
                ]
            )

        embeds = self._embed_tokens(tokens, uncond_mask=uncond_mask)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2, dtype=embeds.dtype)  # merge
        if continuous_segments is not None:
            continuous_segments = self.muq_linear(continuous_segments)
            if uncond_mask is not None:
                uncond_embed = self.unconditional_text_embedding(
                    torch.zeros(1, device=tokens.device, dtype=torch.long)
                )
                mask_expanded = uncond_mask.view(b, 1).expand_as(continuous_segments)
                continuous_segments = torch.where(
                    mask_expanded, uncond_embed, continuous_segments
                )
            batch_indices = torch.arange(h.shape[0], device=h.device)
            h[batch_indices, starts] = continuous_segments
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
        last_h = h[:, -1, :]  # the last frame
        c0_logits = self.codebook0_head(last_h)  # only predict the audio part

        if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
            actual_B = b // 2
            cond_logits = c0_logits[:actual_B, :]
            uncond_logits = c0_logits[actual_B:, :]
            guided_logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            c0_sample = sample_topk(guided_logits, topk, temperature)
            c0_sample = c0_sample.repeat(
                2, 1
            )  # repeat to both branches to keep alignment
        else:
            c0_sample = sample_topk(c0_logits, topk, temperature)

        c0_embed = self._embed_audio(0, c0_sample)

        self.decoder.reset_caches()
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = (
            torch.arange(0, curr_h.size(1), device=curr_h.device)
            .unsqueeze(0)
            .repeat(curr_h.size(0), 1)
        )
        curr_h = curr_h.to(embeds.dtype)
        for i in range(1, self.config.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(
                self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask
            )
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
                actual_B = b // 2
                cond_ci = ci_logits[:actual_B, :]
                uncond_ci = ci_logits[actual_B:, :]
                guided_ci = uncond_ci + (cond_ci - uncond_ci) * cfg_scale

                ci_sample = sample_topk(guided_ci, topk, temperature)
                ci_sample = ci_sample.repeat(2, 1)
            else:
                ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)
            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    def reset_caches(self):
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_local_audio(self, tokens):
        """the token from 0-30"""
        audio_tokens = tokens + (
            self.config.audio_vocab_size
            * torch.arange(self.config.audio_num_codebooks - 1, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks - 1, -1
        )
        return audio_embeds

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(
        self, tokens: torch.Tensor, uncond_mask: torch.Tensor | None
    ) -> torch.Tensor:
        B, S, _ = tokens.size()
        text_embeds = self.text_embeddings(tokens[:, :, -1])

        if uncond_mask is not None:
            uncond_text_embed = self.unconditional_text_embedding(
                torch.zeros(1, device=tokens.device, dtype=torch.long)
            )
            mask_expanded = uncond_mask.view(B, 1, 1).expand_as(text_embeds)
            text_embeds = torch.where(
                mask_expanded,
                uncond_text_embed,
                text_embeds,
            )

        text_embeds = text_embeds.unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size
            * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )
        return torch.cat([audio_embeds, text_embeds], dim=-2)
