"""HeartMuLa – a music language model for lyrics-and-tag conditioned generation.

This module provides :class:`HeartMuLa`, a HuggingFace-compatible
``PreTrainedModel`` that auto-regressively predicts multi-codebook audio
tokens using a Llama-style backbone transformer and a smaller decoder
transformer.  It also contains factory functions for the various Llama
architecture sizes (300 M, 400 M, 3 B, 7 B) and sampling utilities.
"""

import torch
import torch.nn as nn
from .configuration_heartmula import HeartMuLaConfig
from transformers.modeling_utils import PreTrainedModel
import torch
import torch.nn as nn
import torchtune
from torchtune.models import llama3_2


def llama3_2_3B() -> torchtune.modules.transformer.TransformerDecoder:
    """Create a Llama 3.2 transformer with ~3 B parameters."""
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
    """Create a Llama 3.2 transformer with ~300 M parameters."""
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
    """Create a Llama 3.2 transformer with ~7 B parameters."""
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
    """Create a Llama 3.2 transformer with ~400 M parameters."""
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
    """Strip the token embedding and output projection from a torchtune transformer.

    The embedding and head layers are replaced with ``nn.Identity`` so that
    :class:`HeartMuLa` can supply its own multi-modal embeddings and
    prediction heads.

    Args:
        model: A ``TransformerDecoder`` instance from torchtune.

    Returns:
        A tuple of ``(model, embed_dim)`` where *embed_dim* is the original
        embedding dimensionality.
    """
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    """Create a lower-triangular boolean causal attention mask."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """Index into a pre-computed causal mask at the given positions."""
    r = mask[input_pos, :]
    return r


def _multinomial_sample_one_no_sync(probs):
    """Draw one multinomial sample without triggering a CUDA synchronisation.

    Uses the Gumbel-max trick: divides the probabilities by independent
    exponential random variables and returns the argmax.
    """
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    """Sample a token from *logits* using top-k filtering and temperature scaling.

    Args:
        logits: Raw logits of shape ``(..., vocab_size)``.
        topk: Number of highest-probability tokens to keep.
        temperature: Softmax temperature (higher → more random).

    Returns:
        An integer tensor of sampled token indices.
    """
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


class HeartMuLa(PreTrainedModel):
    """HuggingFace-compatible music language model.

    HeartMuLa auto-regressively generates multi-codebook audio tokens
    conditioned on text prompts (tags and lyrics).  It consists of:

    * A **backbone** Llama transformer that processes the interleaved
      text + audio token sequence and predicts the first codebook.
    * A **decoder** Llama transformer that predicts the remaining
      codebooks for each frame, conditioned on the backbone output and
      previously predicted codebooks.

    Args:
        config: A :class:`HeartMuLaConfig` instance.
    """

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
        """Allocate KV caches for the backbone and decoder transformers.

        This must be called before :meth:`generate_frame`.  Existing caches
        are reset before new ones are created.

        Args:
            max_batch_size: Maximum batch size the caches should support.
        """
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
        """Generate one frame of multi-codebook audio tokens.

        The backbone transformer produces the first-codebook logits.  The
        decoder then iteratively predicts each subsequent codebook,
        conditioned on the backbone's last hidden state and the tokens
        sampled so far.

        When ``cfg_scale > 1.0`` and the batch is doubled (conditional +
        unconditional), classifier-free guidance is applied to the logits
        before sampling.

        Args:
            tokens: Input token tensor of shape ``(B, S, num_codebooks+1)``.
            tokens_mask: Boolean mask of shape ``(B, S, num_codebooks+1)``
                indicating which token slots are active.
            input_pos: Positional indices of shape ``(B, S)`` for the KV
                cache.
            temperature: Softmax temperature for sampling.
            topk: Number of top-k candidates to keep when sampling.
            cfg_scale: Classifier-free guidance scale.  Set to ``1.0`` to
                disable guidance.
            continuous_segments: Optional MUQ conditioning embedding of
                shape ``(B, muq_dim)`` inserted at positions *starts*.
            starts: Position index (or indices) at which to insert the
                continuous MUQ embedding.

        Returns:
            An integer tensor of shape ``(B, num_codebooks)`` containing
            the sampled audio tokens for this frame.
        """
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
        """Reset the KV caches of both backbone and decoder transformers."""
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_local_audio(self, tokens):
        """Embed audio tokens from codebooks 1 through N-1 (local codebooks)."""
        audio_tokens = tokens + (
            self.config.audio_vocab_size
            * torch.arange(self.config.audio_num_codebooks - 1, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks - 1, -1
        )
        return audio_embeds

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        """Look up the embedding for *tokens* in the given *codebook* index."""
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(
        self, tokens: torch.Tensor, uncond_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Embed a combined audio + text token tensor.

        The last channel of *tokens* is treated as the text token; all
        preceding channels are audio codebook tokens.  When *uncond_mask*
        is provided, the text embeddings for unconditional (masked) batch
        elements are replaced with a learned unconditional embedding.

        Returns:
            A tensor of shape ``(B, S, num_codebooks+1, embed_dim)``.
        """
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
