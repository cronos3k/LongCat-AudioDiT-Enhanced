"""
LongCat-AudioDiT Enhanced – Gradio Web UI

Primary workflow: Voice Cloning
  1. Upload reference audio → auto-transcribe with Whisper
  2. Type text to synthesise in the cloned voice
  3. Generate → save to Voice Library with a name
  4. Reuse any saved voice from the dropdown

All actions are exposed as Gradio REST API endpoints.

Usage:
    python app.py
    python app.py --port 7860 --share
    python app.py --device cpu
"""

import argparse
import logging
import os
import socket
import time
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from utils import normalize_text, load_audio, approx_duration_from_text
from memory_manager import ModelMemoryManager
from voice_library import get_library
from download_models import (
    download_audiodit, download_whisper,
    _audiodit_present, _whisper_present,
    AUDIODIT_MODELS, WHISPER_MODELS,
    AUDIODIT_DIR, WHISPER_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Memory manager
# ---------------------------------------------------------------------------
_mgr: ModelMemoryManager = None

def get_manager(mode: str = "auto") -> ModelMemoryManager:
    global _mgr
    if _mgr is None or _mgr.mode.value != mode:
        if _mgr is not None:
            _mgr.release_all()
        _mgr = ModelMemoryManager(mode=mode)
    return _mgr

# ---------------------------------------------------------------------------
# Port helpers
# ---------------------------------------------------------------------------
def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) != 0

def find_free_port(start: int = 7860, end: int = 7960) -> int:
    for p in range(start, end):
        if _port_free(p):
            return p
    raise RuntimeError(f"No free port found in {start}-{end}")

# ---------------------------------------------------------------------------
# Core: transcribe reference audio
# ---------------------------------------------------------------------------
def transcribe_reference(audio_path, whisper_size: str, language: str, memory_mode: str, device: str):
    """
    Transcribe a reference audio file with Whisper.
    Returns (transcription_text, status_msg).
    """
    if audio_path is None:
        return "", "Upload a reference audio file first."

    mgr = get_manager(memory_mode)
    try:
        whisper = mgr.get_whisper(whisper_size=whisper_size)
    except Exception as e:
        return "", f"Failed to load Whisper: {e}"

    lang_arg = language if language and language != "auto" else None
    try:
        text, detected = whisper.transcribe(str(audio_path), language=lang_arg)
    except Exception as e:
        return "", f"Transcription failed: {e}"

    return text, f"Transcribed [{detected}] — {len(text)} characters"


# ---------------------------------------------------------------------------
# Core: clone voice (reference audio + transcription → new speech)
# ---------------------------------------------------------------------------
def clone_voice(
    text: str,
    ref_audio_path,
    ref_transcription: str,
    audiodit_size: str,
    nfe: int,
    guidance_strength: float,
    guidance_method: str,
    seed: int,
    memory_mode: str,
    device: str,
):
    """
    Synthesise `text` in the voice captured from `ref_audio_path`.
    Returns (output_audio_path, status_msg).
    """
    if not text or not text.strip():
        return None, "Enter text to synthesise."
    if ref_audio_path is None:
        return None, "Upload a reference audio file."
    if not ref_transcription or not ref_transcription.strip():
        return None, "Reference transcription is empty. Use 'Auto-Transcribe' first."

    mgr = get_manager(memory_mode)
    try:
        model, tokenizer = mgr.get_tts(audiodit_size=audiodit_size, device=device)
    except Exception as e:
        return None, f"Failed to load TTS model: {e}"

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    sr        = model.config.sampling_rate
    full_hop  = model.config.latent_hop
    max_dur   = model.config.max_wav_duration

    synth_text = normalize_text(text)
    ref_text   = normalize_text(ref_transcription)
    full_text  = f"{ref_text} {synth_text}"

    inputs = tokenizer([full_text], padding="longest", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Encode reference audio to get prompt duration
    try:
        off = 3
        pw = load_audio(str(ref_audio_path), sr)
        if pw.shape[-1] % full_hop != 0:
            pw = F.pad(pw, (0, full_hop - pw.shape[-1] % full_hop))
        pw_padded = F.pad(pw, (0, full_hop * off))
        with torch.no_grad():
            plt = model.vae.encode(pw_padded.unsqueeze(0).to(device))
        if off:
            plt = plt[..., :-off]
        prompt_dur = plt.shape[-1]
        prompt_wav = load_audio(str(ref_audio_path), sr).unsqueeze(0)
    except Exception as e:
        return None, f"Failed to process reference audio: {e}"

    prompt_time = prompt_dur * full_hop / sr
    dur_sec = approx_duration_from_text(synth_text, max_duration=max_dur - prompt_time)
    try:
        approx_pd = approx_duration_from_text(ref_text, max_duration=max_dur)
        ratio = np.clip(prompt_time / approx_pd, 1.0, 1.5)
        dur_sec = dur_sec * ratio
    except Exception:
        pass

    duration = int(dur_sec * sr // full_hop)
    duration = min(duration + prompt_dur, int(max_dur * sr // full_hop))

    try:
        with torch.no_grad():
            output = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                prompt_audio=prompt_wav,
                duration=duration,
                steps=nfe,
                cfg_strength=guidance_strength,
                guidance_method=guidance_method,
            )
    except Exception as e:
        return None, f"Generation failed: {e}"

    wav = output.waveform.squeeze().detach().cpu().numpy()
    out_path = OUTPUT_DIR / f"clone_{int(time.time())}.wav"
    sf.write(str(out_path), wav, sr)

    return str(out_path), f"Done — {len(wav)/sr:.2f}s generated"


# ---------------------------------------------------------------------------
# Core: plain TTS (no reference voice)
# ---------------------------------------------------------------------------
def plain_tts(
    text: str,
    audiodit_size: str,
    nfe: int,
    guidance_strength: float,
    guidance_method: str,
    seed: int,
    memory_mode: str,
    device: str,
):
    """Synthesise text with no voice reference (random voice)."""
    if not text or not text.strip():
        return None, "Enter text to synthesise."

    mgr = get_manager(memory_mode)
    try:
        model, tokenizer = mgr.get_tts(audiodit_size=audiodit_size, device=device)
    except Exception as e:
        return None, f"Failed to load TTS model: {e}"

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    sr       = model.config.sampling_rate
    full_hop = model.config.latent_hop
    max_dur  = model.config.max_wav_duration

    t = normalize_text(text)
    inputs = tokenizer([t], padding="longest", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    dur_sec  = approx_duration_from_text(t, max_duration=max_dur)
    duration = int(dur_sec * sr // full_hop)
    duration = min(duration, int(max_dur * sr // full_hop))

    try:
        with torch.no_grad():
            output = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                prompt_audio=None,
                duration=duration,
                steps=nfe,
                cfg_strength=guidance_strength,
                guidance_method=guidance_method,
            )
    except Exception as e:
        return None, f"Generation failed: {e}"

    wav = output.waveform.squeeze().detach().cpu().numpy()
    out_path = OUTPUT_DIR / f"tts_{int(time.time())}.wav"
    sf.write(str(out_path), wav, sr)
    return str(out_path), f"Done — {len(wav)/sr:.2f}s generated"


# ---------------------------------------------------------------------------
# Voice Library helpers (called from UI)
# ---------------------------------------------------------------------------
def library_names_with_placeholder() -> list[str]:
    lib = get_library()
    names = lib.names()
    return ["— select saved voice —"] + names

def save_voice_to_library(name: str, audio_path, transcription: str):
    """Save a (audio, transcription) pair to the library. Returns (new_dropdown, status)."""
    name = (name or "").strip()
    if not name:
        return gr.update(), "Enter a name for this voice."
    if audio_path is None:
        return gr.update(), "No reference audio to save."
    if not transcription or not transcription.strip():
        return gr.update(), "Transcription is empty — auto-transcribe first."
    try:
        get_library().add(name, str(audio_path), transcription)
    except Exception as e:
        return gr.update(), f"Save failed: {e}"
    choices = library_names_with_placeholder()
    return gr.update(choices=choices, value=name), f"Saved '{name}' to voice library."

def load_voice_from_library(name: str):
    """Load a saved voice. Returns (audio_path, transcription, status)."""
    if not name or name.startswith("—"):
        return None, "", ""
    entry = get_library().get(name)
    if entry is None:
        return None, "", f"Voice '{name}' not found."
    audio = entry["audio_path"]
    if not Path(audio).exists():
        return None, "", f"Audio file missing: {audio}"
    return audio, entry["transcription"], f"Loaded '{name}'"

def delete_voice_from_library(name: str):
    """Delete a voice. Returns (new_dropdown_update, status)."""
    if not name or name.startswith("—"):
        return gr.update(), "Select a voice to delete."
    ok = get_library().remove(name)
    choices = library_names_with_placeholder()
    msg = f"Deleted '{name}'." if ok else f"Voice '{name}' not found."
    return gr.update(choices=choices, value=choices[0]), msg

def refresh_library_dropdown():
    choices = library_names_with_placeholder()
    return gr.update(choices=choices)

def library_summary():
    return get_library().summary_text()


# ---------------------------------------------------------------------------
# Status / unload
# ---------------------------------------------------------------------------
def get_status(memory_mode: str) -> str:
    return get_manager(memory_mode).status_str()

def unload_all(memory_mode: str) -> str:
    mgr = get_manager(memory_mode)
    mgr.release_all()
    return "All models unloaded.\n" + mgr.status_str()


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _model_inventory() -> str:
    lines = ["AudioDiT TTS models:"]
    for k, (repo, hint) in AUDIODIT_MODELS.items():
        st = "[downloaded]" if _audiodit_present(k) else "not downloaded"
        lines.append(f"  AudioDiT-{k:<6}  {hint:<8}  {st}")
    lines.append("")
    lines.append("Whisper STT models:")
    for k, (repo, hint) in WHISPER_MODELS.items():
        st = "[downloaded]" if _whisper_present(k) else "not downloaded"
        lines.append(f"  Whisper-{k:<10}  {hint:<8}  {st}")
    return "\n".join(lines)

def download_with_progress(selected_models: list):
    if not selected_models:
        yield "Nothing selected."
        return
    log = []
    def emit(msg):
        log.append(msg)
    for label in selected_models:
        if label.startswith("AudioDiT-"):
            size = label.replace("AudioDiT-", "")
            _, hint = AUDIODIT_MODELS.get(size, ("", "?"))
            log.append(f"AudioDiT-{size} ({hint}): {'already downloaded' if _audiodit_present(size) else 'downloading...'}"); yield "\n".join(log)
            download_audiodit(size, callback=emit); yield "\n".join(log)
        elif label.startswith("Whisper-"):
            size = label.replace("Whisper-", "")
            _, hint = WHISPER_MODELS.get(size, ("", "?"))
            log.append(f"Whisper-{size} ({hint}): {'already downloaded' if _whisper_present(size) else 'downloading...'}"); yield "\n".join(log)
            download_whisper(size, callback=emit); yield "\n".join(log)
    log.extend(["", _model_inventory()])
    yield "\n".join(log)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui(default_device: str = "cuda"):

    AUDIODIT_CHOICES = ["1B", "3.5B"]
    WHISPER_CHOICES  = ["turbo", "large-v3", "medium", "small"]
    MEMORY_MODES     = ["auto", "simultaneous", "sequential"]
    GUIDANCE_METHODS = ["cfg", "apg"]
    LANGUAGE_CHOICES = [
        "auto", "en", "zh", "ja", "ko", "de", "fr", "es", "pt", "ru",
        "ar", "hi", "it", "nl", "pl", "tr", "uk", "vi", "id", "th",
    ]

    with gr.Blocks(title="LongCat-AudioDiT — Voice Cloning") as demo:

        gr.Markdown(
            "# LongCat-AudioDiT — Voice Cloning Studio\n"
            "State-of-the-art voice cloning: give it a reference audio, type your text, get the result."
        )

        # ── Global settings row ──────────────────────────────────────────
        with gr.Row():
            memory_mode_dd = gr.Dropdown(MEMORY_MODES, value="auto", label="Memory Mode", scale=1)
            device_dd      = gr.Dropdown(["cuda", "cpu"], value=default_device, label="Device", scale=1)
            status_box     = gr.Textbox(label="Model Status", lines=3, interactive=False, scale=3)
            with gr.Column(scale=1, min_width=160):
                btn_status = gr.Button("Refresh Status", size="sm")
                btn_unload = gr.Button("Unload All", size="sm", variant="stop")

        gr.Markdown("---")

        with gr.Tabs():

            # ================================================================
            # TAB 1 — Voice Cloning (primary workflow)
            # ================================================================
            with gr.Tab("Voice Cloning"):

                with gr.Row():

                    # ── Left: reference voice ────────────────────────────
                    with gr.Column(scale=2):
                        gr.Markdown("### Reference Voice")

                        with gr.Row():
                            voice_dd = gr.Dropdown(
                                choices=library_names_with_placeholder(),
                                value="— select saved voice —",
                                label="Saved Voices",
                                scale=3,
                            )
                            btn_load_voice   = gr.Button("Load", size="sm", scale=1)
                            btn_refresh_lib  = gr.Button("Refresh", size="sm", scale=1)

                        ref_audio = gr.Audio(
                            label="Reference Audio  (upload or record)",
                            type="filepath",
                        )

                        whisper_dd = gr.Dropdown(
                            WHISPER_CHOICES, value="turbo",
                            label="Whisper Model for Auto-Transcribe",
                        )
                        lang_dd = gr.Dropdown(
                            LANGUAGE_CHOICES, value="auto", label="Language (auto=detect)"
                        )
                        btn_transcribe = gr.Button("Auto-Transcribe Reference", variant="secondary")

                        ref_transcription = gr.Textbox(
                            label="Reference Transcription  (auto-filled or type manually)",
                            lines=3,
                            placeholder="What is being said in the reference audio?",
                        )

                        gr.Markdown("**Save this voice to library**")
                        with gr.Row():
                            voice_name_input = gr.Textbox(
                                label="Voice Name", placeholder="e.g. Alice", scale=3
                            )
                            btn_save_voice  = gr.Button("Save Voice", size="sm", scale=1, variant="primary")
                            btn_delete_voice = gr.Button("Delete", size="sm", scale=1, variant="stop")

                        lib_status = gr.Textbox(
                            label="Library", lines=4, interactive=False,
                            value=library_summary(),
                        )

                    # ── Right: synthesis ─────────────────────────────────
                    with gr.Column(scale=3):
                        gr.Markdown("### Text to Synthesise")

                        synth_text = gr.Textbox(
                            label="Text",
                            lines=6,
                            placeholder="Type what you want spoken in the reference voice…",
                        )

                        with gr.Row():
                            audiodit_dd  = gr.Dropdown(AUDIODIT_CHOICES, value="1B", label="AudioDiT Model")
                            guidance_dd  = gr.Dropdown(GUIDANCE_METHODS, value="cfg", label="Guidance")

                        with gr.Accordion("Advanced", open=False):
                            with gr.Row():
                                nfe_sl       = gr.Slider(4, 64, value=16, step=1, label="ODE Steps")
                                strength_sl  = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="Guidance Strength")
                            seed_nb = gr.Number(value=1024, label="Seed", precision=0)

                        btn_clone = gr.Button(
                            "Generate  —  Clone Voice", variant="primary", size="lg"
                        )

                        clone_audio_out = gr.Audio(label="Output", type="filepath")
                        clone_status    = gr.Textbox(label="Status", lines=2, interactive=False)

                # ── Wire up Tab 1 ────────────────────────────────────────

                btn_transcribe.click(
                    fn=transcribe_reference,
                    inputs=[ref_audio, whisper_dd, lang_dd, memory_mode_dd, device_dd],
                    outputs=[ref_transcription, clone_status],
                    api_name="transcribe_reference",
                )

                btn_clone.click(
                    fn=clone_voice,
                    inputs=[
                        synth_text, ref_audio, ref_transcription,
                        audiodit_dd, nfe_sl, strength_sl, guidance_dd,
                        seed_nb, memory_mode_dd, device_dd,
                    ],
                    outputs=[clone_audio_out, clone_status],
                    api_name="clone_voice",
                )

                btn_save_voice.click(
                    fn=save_voice_to_library,
                    inputs=[voice_name_input, ref_audio, ref_transcription],
                    outputs=[voice_dd, lib_status],
                    api_name="save_voice",
                )

                btn_load_voice.click(
                    fn=load_voice_from_library,
                    inputs=[voice_dd],
                    outputs=[ref_audio, ref_transcription, clone_status],
                    api_name="load_voice",
                )

                btn_delete_voice.click(
                    fn=delete_voice_from_library,
                    inputs=[voice_dd],
                    outputs=[voice_dd, lib_status],
                    api_name="delete_voice",
                )

                btn_refresh_lib.click(
                    fn=lambda: (refresh_library_dropdown(), library_summary()),
                    inputs=[],
                    outputs=[voice_dd, lib_status],
                    api_name="list_voices",
                )

            # ================================================================
            # TAB 2 — Plain TTS (no reference voice)
            # ================================================================
            with gr.Tab("Plain TTS"):
                gr.Markdown(
                    "Synthesise speech without a reference voice. "
                    "The model picks a random voice — useful for testing or when you just need audio."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        tts_text = gr.Textbox(label="Text", lines=6, placeholder="Enter text here…")
                        with gr.Row():
                            tts_model_dd   = gr.Dropdown(AUDIODIT_CHOICES, value="1B", label="Model")
                            tts_guidance_dd = gr.Dropdown(GUIDANCE_METHODS, value="cfg", label="Guidance")
                        with gr.Accordion("Advanced", open=False):
                            with gr.Row():
                                tts_nfe      = gr.Slider(4, 64, value=16, step=1, label="ODE Steps")
                                tts_guidance = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="Guidance Strength")
                            tts_seed = gr.Number(value=1024, label="Seed", precision=0)
                        tts_btn = gr.Button("Generate Speech", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        tts_audio_out = gr.Audio(label="Output", type="filepath")
                        tts_status    = gr.Textbox(label="Status", lines=2, interactive=False)

                tts_btn.click(
                    fn=plain_tts,
                    inputs=[
                        tts_text, tts_model_dd, tts_nfe, tts_guidance,
                        tts_guidance_dd, tts_seed, memory_mode_dd, device_dd,
                    ],
                    outputs=[tts_audio_out, tts_status],
                    api_name="plain_tts",
                )

            # ================================================================
            # TAB 3 — Transcribe Only
            # ================================================================
            with gr.Tab("Transcribe Audio"):
                gr.Markdown("Transcribe any audio file with Whisper — output is plain text.")
                with gr.Row():
                    with gr.Column():
                        stt_audio_in  = gr.Audio(label="Audio", type="filepath")
                        stt_model_dd  = gr.Dropdown(WHISPER_CHOICES, value="turbo", label="Whisper Model")
                        stt_lang_dd   = gr.Dropdown(LANGUAGE_CHOICES, value="auto", label="Language")
                        stt_btn       = gr.Button("Transcribe", variant="primary", size="lg")
                    with gr.Column():
                        stt_text_out  = gr.Textbox(label="Transcription", lines=10)
                        stt_lang_out  = gr.Textbox(label="Detected Language", scale=1)
                        stt_status    = gr.Textbox(label="Status", lines=2, interactive=False)

                stt_btn.click(
                    fn=_stt_flat,
                    inputs=[stt_audio_in, stt_model_dd, stt_lang_dd, memory_mode_dd, device_dd],
                    outputs=[stt_text_out, stt_lang_out, stt_status],
                    api_name="transcribe",
                )

            # ================================================================
            # TAB 4 — Download Models
            # ================================================================
            with gr.Tab("Download Models"):
                gr.Markdown(
                    "**Download models before using them.** "
                    "Select what you need, hit Download, watch the live log. "
                    "Already-downloaded models are skipped automatically."
                )

                _dl_choices = (
                    [f"AudioDiT-{k}  ({hint})" for k, (_, hint) in AUDIODIT_MODELS.items()]
                    + [f"Whisper-{k}  ({hint})" for k, (_, hint) in WHISPER_MODELS.items()]
                )
                _dl_values = (
                    [f"AudioDiT-{k}" for k in AUDIODIT_MODELS]
                    + [f"Whisper-{k}" for k in WHISPER_MODELS]
                )
                _label_to_value = dict(zip(_dl_choices, _dl_values))

                dl_checkboxes = gr.CheckboxGroup(
                    choices=_dl_choices,
                    value=[_dl_choices[0], _dl_choices[2]],
                    label="Models to Download",
                )
                with gr.Row():
                    dl_btn     = gr.Button("Download Selected", variant="primary", size="lg")
                    dl_refresh = gr.Button("Refresh Status", size="lg")

                dl_log = gr.Textbox(
                    label="Download Log", lines=16, interactive=False,
                    value=_model_inventory(),
                )

                def _run_download(selected_labels):
                    keys = [_label_to_value.get(lbl, lbl.split("  ")[0]) for lbl in selected_labels]
                    yield from download_with_progress(keys)

                dl_btn.click(fn=_run_download, inputs=[dl_checkboxes], outputs=[dl_log])
                dl_refresh.click(fn=lambda: _model_inventory(), inputs=[], outputs=[dl_log])

            # ================================================================
            # TAB 5 — About
            # ================================================================
            with gr.Tab("About"):
                gr.Markdown("""
## LongCat-AudioDiT Enhanced

Enhanced fork of [LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT) (Meituan) — Apache-2.0.

### API Endpoints (Gradio REST API)
All actions are available as REST endpoints at `/api/`:

| Endpoint | Description |
|---|---|
| `POST /api/clone_voice` | Clone a voice: text + reference audio + transcription → audio |
| `POST /api/transcribe_reference` | Transcribe reference audio with Whisper |
| `POST /api/plain_tts` | Generate speech without a reference voice |
| `POST /api/transcribe` | Transcribe any audio file |
| `POST /api/save_voice` | Save a voice to the library |
| `POST /api/load_voice` | Load a voice from the library by name |
| `POST /api/delete_voice` | Delete a voice from the library |
| `POST /api/list_voices` | List all saved voices |

### Models
| Model | VRAM | Notes |
|---|---|---|
| AudioDiT-1B | ~4 GB | Fast, great quality |
| AudioDiT-3.5B | ~10 GB | SOTA quality |
| Whisper Turbo | ~1.6 GB | Fast transcription |
| Whisper large-v3 | ~3 GB | Most accurate |

### Voice Library
Voices are stored in `./voices/library.json` with audio files in `./voices/`.
                """)

        # ── Global callbacks ─────────────────────────────────────────────
        btn_status.click(fn=get_status, inputs=[memory_mode_dd], outputs=[status_box])
        btn_unload.click(fn=unload_all, inputs=[memory_mode_dd], outputs=[status_box])
        memory_mode_dd.change(fn=get_status, inputs=[memory_mode_dd], outputs=[status_box])

    return demo


# ---------------------------------------------------------------------------
# STT flat helper (avoids walrus-operator gymnastics in the lambda above)
# ---------------------------------------------------------------------------
def _stt_flat(audio_path, whisper_size, language, memory_mode, device):
    """Returns (transcription, detected_language, status_msg) — three separate values."""
    from memory_manager import ModelMemoryManager
    mgr = get_manager(memory_mode)
    try:
        whisper = mgr.get_whisper(whisper_size=whisper_size)
    except Exception as e:
        return "", "", f"Failed to load Whisper: {e}"
    if audio_path is None:
        return "", "", "Upload an audio file."
    lang_arg = language if language and language != "auto" else None
    try:
        text, detected = whisper.transcribe(str(audio_path), language=lang_arg)
    except Exception as e:
        return "", "", f"Transcription failed: {e}"
    return text, detected, f"Transcribed [{detected}] — {len(text)} chars"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LongCat-AudioDiT Voice Cloning Studio")
    parser.add_argument("--port",   type=int,  default=0)
    parser.add_argument("--host",   type=str,  default="0.0.0.0")
    parser.add_argument("--share",  action="store_true")
    parser.add_argument("--device", type=str,  default="auto")
    parser.add_argument("--mode",   type=str,  default="auto",
                        choices=["auto", "simultaneous", "sequential"])
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    if args.port == 0:
        port = find_free_port(7860, 7960)
    elif not _port_free(args.port):
        logger.warning("Port %d busy, searching…", args.port)
        port = find_free_port(args.port + 1, args.port + 100)
    else:
        port = args.port

    logger.info("Starting on %s:%d (device=%s, mode=%s)", args.host, port, device, args.mode)
    get_manager(args.mode)

    demo = build_ui(default_device=device)
    demo.launch(
        server_name=args.host,
        server_port=port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
