"""
Whisper STT helper for LongCat-AudioDiT.

Supports:
  - faster-whisper backend (CTranslate2, recommended)
  - Model variants: large-v3-turbo ("turbo"), large-v3 ("large-v3")

Usage:
    helper = WhisperHelper(model_size="turbo", device="cuda")
    text, language = helper.transcribe("audio.wav")
    helper.unload()
"""

import gc
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Approximate VRAM usage in GB (fp16 / int8)
WHISPER_VRAM_MAP = {
    "turbo": 1.6,       # large-v3-turbo
    "large-v3": 3.0,
    "medium": 1.5,
    "small": 0.5,
    "base": 0.3,
}

# HuggingFace model IDs for faster-whisper
FASTER_WHISPER_MODELS = {
    "turbo":    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "medium":   "Systran/faster-whisper-medium",
    "small":    "Systran/faster-whisper-small",
    "base":     "Systran/faster-whisper-base",
}


class WhisperHelper:
    """Thin wrapper around faster-whisper for on-demand STT."""

    def __init__(
        self,
        model_size: str = "turbo",
        device: str = "auto",
        compute_type: str = "auto",
        download_root: Optional[str] = None,
    ):
        """
        Args:
            model_size: "turbo", "large-v3", "medium", "small", "base"
            device: "auto", "cuda", "cpu"
            compute_type: "auto", "float16", "int8_float16", "int8"
            download_root: where to cache models (defaults to ./models/whisper/)
        """
        self.model_size = model_size
        self._model = None
        self._is_loaded = False

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Resolve compute type
        if compute_type == "auto":
            if self.device == "cuda":
                self.compute_type = "float16"
            else:
                self.compute_type = "int8"
        else:
            self.compute_type = compute_type

        # Download root: always local to project, never user dirs
        if download_root is None:
            self.download_root = str(Path(__file__).parent / "models" / "whisper")
        else:
            self.download_root = download_root

        Path(self.download_root).mkdir(parents=True, exist_ok=True)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self._model is not None

    @property
    def vram_estimate_gb(self) -> float:
        return WHISPER_VRAM_MAP.get(self.model_size, 3.0)

    def load(self) -> None:
        """Load Whisper model into VRAM/RAM."""
        if self.is_loaded:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            )

        model_id = FASTER_WHISPER_MODELS.get(self.model_size, self.model_size)
        logger.info(
            "Loading Whisper %s on %s (%s) from %s",
            self.model_size, self.device, self.compute_type, model_id,
        )
        self._model = WhisperModel(
            model_id,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.download_root,
        )
        self._is_loaded = True
        logger.info("Whisper %s loaded.", self.model_size)

    def unload(self) -> None:
        """Release VRAM/RAM used by the model."""
        if not self.is_loaded:
            return
        del self._model
        self._model = None
        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Whisper %s unloaded.", self.model_size)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
        auto_load: bool = True,
    ) -> Tuple[str, str]:
        """
        Transcribe an audio file.

        Args:
            audio_path: path to audio file (wav, mp3, flac, …)
            language: ISO 639-1 code ("en", "zh", …) or None for auto-detect
            task: "transcribe" or "translate" (translate → English)
            beam_size: beam search width
            vad_filter: apply voice activity detection filter
            auto_load: load model if not already loaded

        Returns:
            (transcription_text, detected_language)
        """
        if not self.is_loaded:
            if auto_load:
                self.load()
            else:
                raise RuntimeError("Whisper model not loaded. Call load() first.")

        segments, info = self._model.transcribe(
            audio_path,
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        text_parts = [seg.text for seg in segments]
        full_text = " ".join(text_parts).strip()
        detected_lang = info.language

        logger.info(
            "Transcribed %s: '%s...' [lang=%s, prob=%.2f]",
            Path(audio_path).name,
            full_text[:60],
            detected_lang,
            info.language_probability,
        )
        return full_text, detected_lang

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "unloaded"
        return f"WhisperHelper(size={self.model_size}, device={self.device}, {status})"
