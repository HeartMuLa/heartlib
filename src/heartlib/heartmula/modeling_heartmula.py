import torch
import torch.nn as nn
from .configuration_heartmula import HeartMuLaConfig
from transformers.modeling_utils import PreTrainedModel
import torch
import torch.nn as nn
import torchtune
from torchtune.models import llama3_2
from torchtune.modules import KVCache
from torchtune.modules.common_utils import delete_kv_caches
from typing import Optional, Tuple


class _PrefixKVCache(KVCache):
    """A KVCache that hands attention only the positions actually written.

    torchtune's ``KVCache.update`` returns the whole cache tensor, so a decode
    step attends over every *allocated* position and leans on the causal mask
    to throw the unwritten tail away. Cost is therefore pinned at the worst
    case from the very first frame. Returning a view of the filled prefix makes
    it grow with how far into the song we actually are.

    The fill level is tracked as a Python int rather than read back from
    ``cache_pos``, which lives on the GPU: reading that would force a device
    sync in each of the backbone's 28 layers, on every frame.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filled = 0

    def reset(self) -> None:
        super().reset()
        self.filled = 0

    def update(
        self, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_out, v_out = super().update(k_val, v_val)
        self.filled += k_val.shape[2]
        return k_out[:, :, : self.filled], v_out[:, :, : self.filled]


def _install_prefix_caches(model) -> None:
    """Swap the caches torchtune just built for prefix-slicing ones."""
    for layer in model.layers:
        old = layer.attn.kv_cache
        if old is None or isinstance(old, _PrefixKVCache):
            continue
        batch_size, num_heads, max_seq_len, head_dim = old.k_cache.shape
        layer.attn.kv_cache = _PrefixKVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=old.k_cache.dtype,
        ).to(old.k_cache.device)


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
    return model, embed_dim


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


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

    def setup_caches(self, max_batch_size: int, max_seq_len: Optional[int] = None):
        """Allocate the KV caches.

        Args:
            max_batch_size: Batch size the caches must hold.
            max_seq_len: Longest position the backbone will be asked to attend
                to, i.e. prompt length plus the number of frames to generate.
                Defaults to the model's full context.

        Sizing this to the actual request matters a lot. torchtune's KVCache
        hands the *whole* cache tensor to attention rather than a view of the
        filled prefix, so every decode step reads all `max_seq_len` positions
        and softmaxes over them, only to have the causal mask discard the
        unwritten tail. On a 2.8B backbone that is 1.75 GB of KV traffic per
        frame at the full 8192 context. Measured on gfx1151, one backbone
        forward: 248 ms at 8192, 150 ms at 4096, 75 ms at 1024. The output is
        unchanged -- the positions dropped were already masked out.
        """
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        backbone_max_seq_len = self.backbone.max_seq_len
        if max_seq_len is not None:
            # Round up: a ragged cache length buys nothing and complicates
            # nothing, but a tidy one keeps kernel shapes predictable.
            requested = min(_round_up(max_seq_len, 128), self.backbone.max_seq_len)
            backbone_max_seq_len = max(requested, 128)

        # reset_caches() only zeroes existing caches; it cannot resize them, and
        # torchtune refuses (with a warning, not an error) to set up caches that
        # already exist. Delete them so a new length actually takes effect.
        if self.backbone.caches_are_enabled():
            delete_kv_caches(self.backbone)
        if self.decoder.caches_are_enabled():
            delete_kv_caches(self.decoder)

        with device:
            self.backbone.setup_caches(
                max_batch_size, dtype, decoder_max_seq_len=backbone_max_seq_len
            )
            # Backbone only: the decoder's cache is 8 positions deep, so there
            # is nothing to save there.
            _install_prefix_caches(self.backbone)
            self.decoder.setup_caches(
                max_batch_size,
                dtype,
                decoder_max_seq_len=self.config.audio_num_codebooks,
            )

        self.register_buffer(
            "backbone_causal_mask",
            _create_causal_mask(backbone_max_seq_len, device),
        )
        self.register_buffer(
            "decoder_causal_mask",
            _create_causal_mask(self.config.audio_num_codebooks, device),
        )

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
        # The prefix cache will hand attention only the first `attended`
        # positions, so the mask has to be narrowed to match.
        curr_backbone_mask = self.backbone_causal_mask[
            input_pos, : self._backbone_attended(s)
        ]

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

    def _backbone_attended(self, seq_len: int) -> int:
        """Number of cache positions this forward will actually attend over."""
        cache = self.backbone.layers[0].attn.kv_cache
        if isinstance(cache, _PrefixKVCache):
            return cache.filled + seq_len
        # Stock KVCache returns the whole tensor; the mask must span it.
        return self.backbone_causal_mask.shape[0]

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
