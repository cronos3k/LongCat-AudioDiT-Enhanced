"""
VRAM Memory Manager for LongCat-AudioDiT + Whisper.

Orchestrates loading and unloading of:
  - AudioDiT TTS models (1B / 3.5B)
  - Whisper STT models (turbo / large-v3)

Modes:
  "auto"        – probe available VRAM; keep both loaded if possible, else sequential
  "simultaneous"– always keep both loaded (fails if VRAM too small)
  "sequential"  – always unload one before loading the other (safest for ≤12GB)
"""

import gc
import logging
from enum import Enum
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Estimated peak VRAM (GB) per model in fp16 / int8 on 1 GPU
AUDIODIT_VRAM = {
    "1B":   4.0,
    "3.5B": 10.0,
}
WHISPER_VRAM = {
    "turbo":    1.6,
    "large-v3": 3.0,
}
# Leave this headroom free for activations, KV-cache, OS
VRAM_HEADROOM_GB = 2.0


class LoadMode(str, Enum):
    AUTO        = "auto"
    SIMULTANEOUS = "simultaneous"
    SEQUENTIAL  = "sequential"


def _available_vram_gb() -> float:
    """Return free VRAM in GB on the default CUDA device, or 0 if no GPU."""
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


def _total_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    _, total = torch.cuda.mem_get_info()
    return total / (1024 ** 3)


def _used_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    allocated = torch.cuda.memory_allocated()
    reserved  = torch.cuda.memory_reserved()
    return max(allocated, reserved) / (1024 ** 3)


class ModelMemoryManager:
    """
    Coordinates AudioDiT + Whisper model lifecycle.

    Typical usage::

        mgr = ModelMemoryManager(mode="auto")
        tts_model, tokenizer = mgr.get_tts(audiodit_size="1B", device="cuda")
        # ... generate audio ...

        whisper = mgr.get_whisper(whisper_size="turbo")
        text, lang = whisper.transcribe("audio.wav")

        mgr.release_all()
    """

    def __init__(self, mode: str = "auto"):
        self.mode = LoadMode(mode)
        self._tts_model = None
        self._tts_tokenizer = None
        self._tts_size: Optional[str] = None
        self._tts_device: Optional[str] = None

        self._whisper: Optional[object] = None  # WhisperHelper
        self._whisper_size: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tts(self, audiodit_size: str = "1B", device: str = "cuda"):
        """
        Return (AudioDiTModel, tokenizer), loading if necessary.

        If mode is sequential and Whisper is loaded, Whisper is unloaded first.
        """
        if self._tts_model is not None and self._tts_size == audiodit_size:
            return self._tts_model, self._tts_tokenizer

        # Need to load a (potentially different) TTS model
        if self._tts_model is not None:
            self._unload_tts()

        # Sequential: unload Whisper first
        if self._should_unload_whisper_for_tts(audiodit_size):
            logger.info("Sequential mode: unloading Whisper before loading AudioDiT %s", audiodit_size)
            self._unload_whisper()

        self._load_tts(audiodit_size, device)
        return self._tts_model, self._tts_tokenizer

    def get_whisper(self, whisper_size: str = "turbo"):
        """
        Return WhisperHelper, loading if necessary.

        If mode is sequential and TTS is loaded, TTS is unloaded first.
        """
        from whisper_helper import WhisperHelper

        if self._whisper is not None and self._whisper_size == whisper_size:
            return self._whisper

        # Need to (re)load
        if self._whisper is not None:
            self._unload_whisper()

        # Sequential: unload TTS first
        if self._should_unload_tts_for_whisper(whisper_size):
            logger.info("Sequential mode: unloading AudioDiT before loading Whisper %s", whisper_size)
            self._unload_tts()

        device = self._tts_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._whisper = WhisperHelper(model_size=whisper_size, device=device)
        self._whisper.load()
        self._whisper_size = whisper_size
        return self._whisper

    def release_tts(self):
        """Explicitly unload TTS model."""
        self._unload_tts()

    def release_whisper(self):
        """Explicitly unload Whisper model."""
        self._unload_whisper()

    def release_all(self):
        """Unload everything and free VRAM."""
        self._unload_tts()
        self._unload_whisper()

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def status(self) -> Dict:
        tts_loaded = self._tts_model is not None
        whisper_loaded = self._whisper is not None and getattr(self._whisper, "is_loaded", False)
        return {
            "mode": self.mode.value,
            "tts_loaded": tts_loaded,
            "tts_size": self._tts_size if tts_loaded else None,
            "whisper_loaded": whisper_loaded,
            "whisper_size": self._whisper_size if whisper_loaded else None,
            "vram_used_gb": round(_used_vram_gb(), 2),
            "vram_total_gb": round(_total_vram_gb(), 2),
            "vram_free_gb": round(_available_vram_gb(), 2),
        }

    def status_str(self) -> str:
        s = self.status()
        lines = [
            f"Mode: {s['mode']}",
            f"TTS:     {'[ON]  ' + s['tts_size'] if s['tts_loaded'] else '[OFF] not loaded'}",
            f"Whisper: {'[ON]  ' + s['whisper_size'] if s['whisper_loaded'] else '[OFF] not loaded'}",
        ]
        if torch.cuda.is_available():
            lines.append(
                f"VRAM:    {s['vram_used_gb']:.1f} / {s['vram_total_gb']:.1f} GB  "
                f"({s['vram_free_gb']:.1f} GB free)"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _should_unload_whisper_for_tts(self, audiodit_size: str) -> bool:
        if self._whisper is None:
            return False
        if self.mode == LoadMode.SEQUENTIAL:
            return True
        if self.mode == LoadMode.SIMULTANEOUS:
            return False
        # AUTO: check if both fit
        needed = AUDIODIT_VRAM.get(audiodit_size, 10.0) + WHISPER_VRAM.get(self._whisper_size, 3.0)
        available = _available_vram_gb() + WHISPER_VRAM.get(self._whisper_size, 3.0)  # pretend whisper free
        return needed + VRAM_HEADROOM_GB > available

    def _should_unload_tts_for_whisper(self, whisper_size: str) -> bool:
        if self._tts_model is None:
            return False
        if self.mode == LoadMode.SEQUENTIAL:
            return True
        if self.mode == LoadMode.SIMULTANEOUS:
            return False
        # AUTO
        needed = AUDIODIT_VRAM.get(self._tts_size, 10.0) + WHISPER_VRAM.get(whisper_size, 3.0)
        available = _available_vram_gb() + AUDIODIT_VRAM.get(self._tts_size, 10.0)
        return needed + VRAM_HEADROOM_GB > available

    def _load_tts(self, audiodit_size: str, device: str):
        import audiodit  # noqa: F401 – registers AutoConfig / AutoModel
        from audiodit import AudioDiTModel
        from transformers import AutoTokenizer
        from pathlib import Path
        from safetensors import safe_open

        # Prefer local model dir; fall back to HF Hub id
        local_dir_map = {
            "1B":   Path(__file__).parent / "models" / "audiodit" / "1B",
            "3.5B": Path(__file__).parent / "models" / "audiodit" / "3.5B",
        }
        hf_id_map = {
            "1B":   "meituan-longcat/LongCat-AudioDiT-1B",
            "3.5B": "meituan-longcat/LongCat-AudioDiT-3.5B",
        }
        local_dir = local_dir_map.get(audiodit_size)
        if local_dir and (local_dir / "config.json").exists():
            model_id = str(local_dir)
            safetensors_path = local_dir / "model.safetensors"
        else:
            model_id = hf_id_map.get(audiodit_size, audiodit_size)
            safetensors_path = None

        logger.info("Loading AudioDiT %s from %s on %s …", audiodit_size, model_id, device)
        torch_device = torch.device(device)
        model = AudioDiTModel.from_pretrained(model_id).to(torch_device)

        # Transformers 5.x uses meta-device init which breaks weight_norm parameters
        # in the VAE (weight_g stays zero → NaN output). Fix: reload VAE weights
        # directly from safetensors, bypassing the meta-device path.
        # When loading from HF Hub, find the cached safetensors file.
        if safetensors_path is None:
            try:
                from huggingface_hub import try_to_load_from_cache
                cached = try_to_load_from_cache(model_id, "model.safetensors")
                if cached and Path(cached).exists():
                    safetensors_path = Path(cached)
            except Exception:
                pass

        if safetensors_path and Path(safetensors_path).exists():
            logger.info("Reloading VAE weights from safetensors (meta-device fix) …")
            vae_sd = {}
            with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith("vae."):
                        vae_sd[k[4:]] = f.get_tensor(k)
            model.vae.load_state_dict(vae_sd, strict=True)
            logger.info("VAE weights reloaded OK.")
        else:
            logger.warning("Could not find safetensors for VAE fix — output may be silence.")

        model.vae.to_half()
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)

        self._tts_model = model
        self._tts_tokenizer = tokenizer
        self._tts_size = audiodit_size
        self._tts_device = device
        logger.info("AudioDiT %s loaded.", audiodit_size)

    def _unload_tts(self):
        if self._tts_model is None:
            return
        logger.info("Unloading AudioDiT %s …", self._tts_size)
        del self._tts_model
        del self._tts_tokenizer
        self._tts_model = None
        self._tts_tokenizer = None
        self._tts_size = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("AudioDiT unloaded.")

    def _unload_whisper(self):
        if self._whisper is None:
            return
        logger.info("Unloading Whisper %s …", self._whisper_size)
        self._whisper.unload()
        self._whisper = None
        self._whisper_size = None
        logger.info("Whisper unloaded.")
