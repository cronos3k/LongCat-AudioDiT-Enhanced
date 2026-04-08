"""
HuggingFace Spaces entry point for LongCat-AudioDiT Enhanced.

Hackathon version:
  - Pre-downloads all models at startup (no download lag during use)
  - Uses ZeroGPU (@spaces.GPU) for on-demand GPU allocation
  - /tmp storage for outputs, models, voices
"""

import os
import sys
import time
from pathlib import Path

# ── Redirect HF cache + writable dirs to /tmp ────────────────────────────────
os.environ["HF_HOME"]            = "/tmp/hf_home"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_home/transformers"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_home/datasets"

for d in ["/tmp/hf_home", "/tmp/audiodit_outputs", "/tmp/audiodit_voices", "/tmp/hf_home/whisper"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ── Pre-download all models at startup ────────────────────────────────────────
from huggingface_hub import snapshot_download

t0 = time.time()

print("[Spaces] Pre-downloading AudioDiT-1B …")
snapshot_download("meituan-longcat/LongCat-AudioDiT-1B")

print("[Spaces] Pre-downloading text encoder (google/umt5-base) …")
snapshot_download("google/umt5-base")

print("[Spaces] Pre-downloading Whisper Turbo …")
snapshot_download(
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    local_dir="/tmp/hf_home/whisper",
)

print(f"[Spaces] All models pre-downloaded in {time.time() - t0:.0f}s")

# ── Patch app constants before import ─────────────────────────────────────────
import app as _app
import voice_library as _vl
import whisper_helper as _wh

_app.OUTPUT_DIR = Path("/tmp/audiodit_outputs")

# Patch voice library to store in /tmp
_vl.VOICES_DIR   = Path("/tmp/audiodit_voices")
_vl.LIBRARY_FILE = Path("/tmp/audiodit_voices/library.json")
_vl.VOICES_DIR.mkdir(parents=True, exist_ok=True)
_vl._library = None

# Patch Whisper download root to /tmp (already pre-downloaded there)
_orig_wh_init = _wh.WhisperHelper.__init__
def _patched_wh_init(self, model_size="turbo", device="auto", compute_type="auto", download_root=None):
    _orig_wh_init(self, model_size=model_size, device=device, compute_type=compute_type,
                  download_root=download_root or "/tmp/hf_home/whisper")
_wh.WhisperHelper.__init__ = _patched_wh_init

# ── ZeroGPU: wrap GPU-needing functions before build_ui references them ───────
import spaces
import torch


_orig_clone_voice = _app.clone_voice

@spaces.GPU(duration=180)
def _gpu_clone_voice(text, ref_audio_path, ref_transcription, audiodit_size, nfe,
                     guidance_strength, guidance_method, seed, memory_mode, device):
    try:
        _app.get_manager(memory_mode).release_all()
    except Exception:
        pass
    return _orig_clone_voice(text, ref_audio_path, ref_transcription, audiodit_size,
                             nfe, guidance_strength, guidance_method, seed,
                             memory_mode, "cuda")

_app.clone_voice = _gpu_clone_voice


_orig_plain_tts = _app.plain_tts

@spaces.GPU(duration=180)
def _gpu_plain_tts(text, audiodit_size, nfe, guidance_strength, guidance_method,
                   seed, memory_mode, device):
    try:
        _app.get_manager(memory_mode).release_all()
    except Exception:
        pass
    return _orig_plain_tts(text, audiodit_size, nfe, guidance_strength, guidance_method,
                           seed, memory_mode, "cuda")

_app.plain_tts = _gpu_plain_tts


_orig_transcribe = _app.transcribe_reference

@spaces.GPU(duration=120)
def _gpu_transcribe(audio_path, whisper_size, language, memory_mode, device):
    try:
        _app.get_manager(memory_mode).release_all()
    except Exception:
        pass
    return _orig_transcribe(audio_path, whisper_size, language, memory_mode, "cuda")

_app.transcribe_reference = _gpu_transcribe


_orig_stt_flat = _app._stt_flat

@spaces.GPU(duration=120)
def _gpu_stt_flat(audio_path, whisper_size, language, memory_mode, device):
    try:
        _app.get_manager(memory_mode).release_all()
    except Exception:
        pass
    return _orig_stt_flat(audio_path, whisper_size, language, memory_mode, "cuda")

_app._stt_flat = _gpu_stt_flat

# ── Launch ────────────────────────────────────────────────────────────────────
import gradio as gr

print(f"[Spaces] ZeroGPU active, CUDA at launch: {torch.cuda.is_available()}")

demo = _app.build_ui(default_device="cuda")
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    share=False,
    show_error=True,
)
