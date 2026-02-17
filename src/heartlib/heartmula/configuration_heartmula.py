"""Configuration for the HeartMuLa music language model."""

from transformers.configuration_utils import PretrainedConfig


class HeartMuLaConfig(PretrainedConfig):
    """Configuration class for :class:`HeartMuLa`.

    HeartMuLa is a music language model that auto-regressively generates
    audio tokens conditioned on text (tags + lyrics).  It uses a Llama-style
    backbone transformer and a smaller decoder transformer for multi-codebook
    audio token prediction.

    Args:
        backbone_flavor: Architecture variant for the backbone transformer.
            Must be a key in the ``FLAVORS`` registry (e.g. ``"llama-3B"``).
        decoder_flavor: Architecture variant for the decoder transformer
            (e.g. ``"llama-300M"``).
        text_vocab_size: Size of the text token vocabulary.
        audio_vocab_size: Size of each audio codebook vocabulary.
        audio_num_codebooks: Number of audio codebooks produced per frame.
        muq_dim: Dimensionality of the MUQ (music query) conditioning
            embedding.
    """

    model_type = "heartmula"

    def __init__(
        self,
        backbone_flavor: str = "llama-3B",
        decoder_flavor: str = "llama-300M",
        text_vocab_size: int = 128256,
        audio_vocab_size: int = 8197,
        audio_num_codebooks: int = 8,
        muq_dim: int = 512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backbone_flavor = backbone_flavor
        self.decoder_flavor = decoder_flavor
        self.text_vocab_size = text_vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.audio_num_codebooks = audio_num_codebooks
        self.muq_dim = muq_dim
