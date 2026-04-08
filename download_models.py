"""
Model downloader for LongCat-AudioDiT Enhanced.

Downloads models to ./models/ so they are available offline.
Always download BEFORE running the GUI - never let the GUI block on a download.

Usage:
    python download_models.py                          # 1B + whisper turbo (recommended start)
    python download_models.py --tts 1B 3.5B --whisper turbo large-v3
    python download_models.py --all
    python download_models.py --list
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
AUDIODIT_MODELS = {
    "1B":   ("meituan-longcat/LongCat-AudioDiT-1B",   "~4 GB"),
    "3.5B": ("meituan-longcat/LongCat-AudioDiT-3.5B", "~10 GB"),
}

WHISPER_MODELS = {
    "turbo":    ("deepdml/faster-whisper-large-v3-turbo-ct2", "~1.6 GB"),
    "large-v3": ("Systran/faster-whisper-large-v3",            "~3 GB"),
    "medium":   ("Systran/faster-whisper-medium",               "~1.5 GB"),
    "small":    ("Systran/faster-whisper-small",                "~0.5 GB"),
}

# Local cache dirs – always project-local, never Windows user dirs
MODELS_DIR   = Path(__file__).parent / "models"
AUDIODIT_DIR = MODELS_DIR / "audiodit"
WHISPER_DIR  = MODELS_DIR / "whisper"


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def _audiodit_present(size: str) -> bool:
    """True only when the weights file exists and is fully written (no .incomplete sibling)."""
    weights = AUDIODIT_DIR / size / "model.safetensors"
    incomplete = AUDIODIT_DIR / size / ".cache" / "huggingface" / "download" / "model.safetensors.incomplete"
    return weights.exists() and not incomplete.exists()

def _whisper_present(size: str) -> bool:
    """True only when the model.bin weights file exists and is fully written."""
    d = WHISPER_DIR / size
    weights = d / "model.bin"
    incomplete = d / ".cache" / "huggingface" / "download" / "model.bin.incomplete"
    return weights.exists() and not incomplete.exists()


def model_status() -> dict:
    """Return a dict with download status for every model."""
    status = {}
    for k in AUDIODIT_MODELS:
        status[f"audiodit_{k}"] = _audiodit_present(k)
    for k in WHISPER_MODELS:
        status[f"whisper_{k}"] = _whisper_present(k)
    return status


# ---------------------------------------------------------------------------
# Progress callback for huggingface_hub
# ---------------------------------------------------------------------------

class _ProgressPrinter:
    """Prints file-level download progress to stdout."""

    def __init__(self, label: str):
        self.label = label
        self._last_print = 0.0
        self._files_done: set = set()

    def __call__(self, info):
        # info is a tqdm-like object from huggingface_hub
        try:
            filename = getattr(info, "filename", "")
            downloaded = getattr(info, "downloaded", 0)
            total      = getattr(info, "total", 0)
            now = time.time()
            if total and now - self._last_print >= 2.0:
                pct = downloaded / total * 100
                mb_done = downloaded / 1e6
                mb_total = total / 1e6
                print(
                    f"\r  [{self.label}] {filename:40s}  "
                    f"{mb_done:7.1f} / {mb_total:7.1f} MB  ({pct:5.1f}%)",
                    end="", flush=True,
                )
                self._last_print = now
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Core download functions
# ---------------------------------------------------------------------------

def download_audiodit(size: str, callback=None) -> bool:
    """Download an AudioDiT model. Returns True on success."""
    entry = AUDIODIT_MODELS.get(size)
    if not entry:
        logger.error("Unknown AudioDiT size '%s'. Choose from: %s", size, list(AUDIODIT_MODELS))
        return False
    repo_id, size_hint = entry
    local_dir = AUDIODIT_DIR / size

    if _audiodit_present(size):
        msg = f"AudioDiT-{size} already downloaded at {local_dir}"
        logger.info(msg)
        if callback:
            callback(msg)
        return True

    local_dir.mkdir(parents=True, exist_ok=True)
    msg = f"Downloading AudioDiT-{size} ({size_hint}) from {repo_id} ..."
    print(f"\n{msg}")
    if callback:
        callback(msg)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
        )
        print()  # newline after progress
        msg = f"[OK] AudioDiT-{size} -> {local_dir}"
        logger.info(msg)
        if callback:
            callback(msg)
        return True
    except Exception as e:
        print()
        msg = f"FAILED to download AudioDiT-{size}: {e}"
        logger.error(msg)
        if callback:
            callback(msg)
        return False


def download_whisper(size: str, callback=None) -> bool:
    """Download a Whisper model. Returns True on success."""
    entry = WHISPER_MODELS.get(size)
    if not entry:
        logger.error("Unknown Whisper size '%s'. Choose from: %s", size, list(WHISPER_MODELS))
        return False
    repo_id, size_hint = entry
    local_dir = WHISPER_DIR / size

    if _whisper_present(size):
        msg = f"Whisper-{size} already downloaded at {local_dir}"
        logger.info(msg)
        if callback:
            callback(msg)
        return True

    local_dir.mkdir(parents=True, exist_ok=True)
    msg = f"Downloading Whisper-{size} ({size_hint}) from {repo_id} ..."
    print(f"\n{msg}")
    if callback:
        callback(msg)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
        )
        print()
        msg = f"[OK] Whisper-{size} -> {local_dir}"
        logger.info(msg)
        if callback:
            callback(msg)
        return True
    except Exception as e:
        print()
        msg = f"FAILED to download Whisper-{size}: {e}"
        logger.error(msg)
        if callback:
            callback(msg)
        return False


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def list_models():
    print("\n  AudioDiT TTS models:")
    print(f"  {'Name':<8}  {'Size':<8}  {'Status':<18}  HuggingFace repo")
    print(f"  {'-'*8}  {'-'*8}  {'-'*18}  {'-'*40}")
    for k, (repo, hint) in AUDIODIT_MODELS.items():
        st = "[downloaded]" if _audiodit_present(k) else "not downloaded"
        print(f"  {k:<8}  {hint:<8}  {st:<18}  {repo}")

    print(f"\n  Whisper STT models:")
    print(f"  {'Name':<10}  {'Size':<8}  {'Status':<18}  HuggingFace repo")
    print(f"  {'-'*10}  {'-'*8}  {'-'*18}  {'-'*45}")
    for k, (repo, hint) in WHISPER_MODELS.items():
        st = "[downloaded]" if _whisper_present(k) else "not downloaded"
        print(f"  {k:<10}  {hint:<8}  {st:<18}  {repo}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download LongCat-AudioDiT + Whisper models to ./models/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_models.py                            # 1B TTS + Whisper Turbo (~6 GB)
  python download_models.py --tts 1B 3.5B             # both TTS models
  python download_models.py --whisper large-v3         # best Whisper only
  python download_models.py --all                      # everything (~19 GB)
  python download_models.py --list                     # show status and exit
        """,
    )
    parser.add_argument("--tts",     nargs="+", metavar="SIZE",
                        help=f"TTS models: {list(AUDIODIT_MODELS)}")
    parser.add_argument("--whisper", nargs="+", metavar="SIZE",
                        help=f"Whisper models: {list(WHISPER_MODELS)}")
    parser.add_argument("--all",  action="store_true", help="Download every model")
    parser.add_argument("--list", action="store_true", help="List status and exit")
    args = parser.parse_args()

    AUDIODIT_DIR.mkdir(parents=True, exist_ok=True)
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_models()
        return

    if args.all:
        tts_sizes     = list(AUDIODIT_MODELS)
        whisper_sizes = list(WHISPER_MODELS)
    else:
        tts_sizes     = args.tts     or ["1B"]
        whisper_sizes = args.whisper or ["turbo"]

    # Show what we're about to do
    print("\n  === LongCat-AudioDiT Model Downloader ===")
    for s in tts_sizes:
        _, hint = AUDIODIT_MODELS.get(s, ("?", "?"))
        status = "[already have it]" if _audiodit_present(s) else f"will download {hint}"
        print(f"  AudioDiT-{s:<6}  {status}")
    for s in whisper_sizes:
        _, hint = WHISPER_MODELS.get(s, ("?", "?"))
        status = "[already have it]" if _whisper_present(s) else f"will download {hint}"
        print(f"  Whisper-{s:<8}  {status}")
    print()

    ok = True
    t0 = time.time()
    for s in tts_sizes:
        ok &= download_audiodit(s)
    for s in whisper_sizes:
        ok &= download_whisper(s)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s.")
    list_models()

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
