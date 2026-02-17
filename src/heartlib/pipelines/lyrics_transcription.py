"""HeartTranscriptor – a Whisper-based lyrics transcription pipeline.

This module wraps a fine-tuned Whisper model for music lyrics transcription
behind the HuggingFace ``AutomaticSpeechRecognitionPipeline`` interface.
"""

from transformers.pipelines.automatic_speech_recognition import (
    AutomaticSpeechRecognitionPipeline,
)
from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration
from transformers.models.whisper.processing_whisper import WhisperProcessor
import torch
import os


class HeartTranscriptorPipeline(AutomaticSpeechRecognitionPipeline):
    """HuggingFace ASR pipeline specialised for music lyrics transcription.

    This thin subclass delegates all heavy lifting to the parent
    ``AutomaticSpeechRecognitionPipeline`` and simply provides a
    :meth:`from_pretrained` factory that knows how to locate the
    HeartTranscriptor checkpoint inside the shared model directory.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls, pretrained_path: str, device: torch.device, dtype: torch.dtype
    ):
        """Load a HeartTranscriptor pipeline from a local checkpoint directory.

        Args:
            pretrained_path: Root directory containing a
                ``HeartTranscriptor-oss/`` sub-folder with the Whisper
                model weights and processor files.
            device: Torch device on which to place the model.
            dtype: Inference dtype (e.g. ``torch.float16``).

        Returns:
            A ready-to-use :class:`HeartTranscriptorPipeline` instance.

        Raises:
            FileNotFoundError: If the expected checkpoint directory is
                missing.
        """
        if os.path.exists(
            hearttranscriptor_path := os.path.join(
                pretrained_path, "HeartTranscriptor-oss"
            )
        ):
            model = WhisperForConditionalGeneration.from_pretrained(
                hearttranscriptor_path, torch_dtype=dtype, low_cpu_mem_usage=True
            )
            processor = WhisperProcessor.from_pretrained(hearttranscriptor_path)
        else:
            raise FileNotFoundError(
                f"Expected to find checkpoint for HeartTranscriptor at {hearttranscriptor_path} but not found. Please check your folder {pretrained_path}."
            )

        return cls(
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            dtype=dtype,
            chunk_length_s=30,
            batch_size=16,
        )
