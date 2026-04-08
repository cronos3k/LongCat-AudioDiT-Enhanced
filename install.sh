#!/usr/bin/env bash
# =============================================================================
# install.sh — LongCat-AudioDiT Enhanced installer for Linux / macOS
# =============================================================================
# Usage:
#   bash install.sh                     # default: CUDA 12.x, Python 3.10+
#   bash install.sh --cpu               # CPU-only build
#   bash install.sh --download-models   # also download models after install
#   bash install.sh --python python3.11 # specify python binary
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
REQ_FILE="$SCRIPT_DIR/requirements_enhanced.txt"

CPU_ONLY=0
DOWNLOAD_MODELS=0
PYTHON_BIN="python3"

# ── Parse arguments ──────────────────────────────────────────────────────────
for arg in "$@"; do
    case $arg in
        --cpu)              CPU_ONLY=1 ;;
        --download-models)  DOWNLOAD_MODELS=1 ;;
        --python=*)         PYTHON_BIN="${arg#*=}" ;;
        --python)           shift; PYTHON_BIN="$1" ;;
        -h|--help)
            echo "Usage: bash install.sh [--cpu] [--download-models] [--python=python3.x]"
            exit 0
            ;;
    esac
done

echo "======================================================================"
echo "  LongCat-AudioDiT Enhanced — Installer"
echo "======================================================================"
echo "  Script dir : $SCRIPT_DIR"
echo "  Venv dir   : $VENV_DIR"
echo "  Python     : $PYTHON_BIN"
echo "  CPU only   : $CPU_ONLY"
echo "======================================================================"

# ── Check Python version ─────────────────────────────────────────────────────
echo ""
echo ">>> Checking Python version …"
$PYTHON_BIN -c "
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 10):
    print(f'ERROR: Python 3.10+ required, found {major}.{minor}')
    sys.exit(1)
print(f'OK: Python {major}.{minor}.{sys.version_info[2]}')
"

# ── Create virtual environment ───────────────────────────────────────────────
echo ""
echo ">>> Creating virtual environment at $VENV_DIR …"
if [ -d "$VENV_DIR" ]; then
    echo "    Venv already exists — reusing."
else
    $PYTHON_BIN -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "    Activated: $(which python)"

# ── Upgrade pip / setuptools ─────────────────────────────────────────────────
echo ""
echo ">>> Upgrading pip …"
pip install --upgrade pip setuptools wheel -q

# ── Install PyTorch ──────────────────────────────────────────────────────────
echo ""
echo ">>> Installing PyTorch …"
if [ "$CPU_ONLY" -eq 1 ]; then
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    # Try CUDA 12.4 first (most modern GPUs), fallback to 12.1
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 || \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# ── Install project requirements ─────────────────────────────────────────────
echo ""
echo ">>> Installing project requirements …"
pip install -r "$REQ_FILE"

# ── Verify key imports ───────────────────────────────────────────────────────
echo ""
echo ">>> Verifying installation …"
python -c "
import torch, torchaudio, transformers, gradio, faster_whisper, soundfile, librosa
print(f'  torch           {torch.__version__}')
print(f'  torchaudio      {torchaudio.__version__}')
print(f'  transformers    {transformers.__version__}')
print(f'  gradio          {gradio.__version__}')
print(f'  cuda available  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU             {torch.cuda.get_device_name(0)}')
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  VRAM            {total_gb:.1f} GB')
print('All imports OK.')
"

# ── Write launch scripts ─────────────────────────────────────────────────────
echo ""
echo ">>> Writing launch scripts …"

cat > "$SCRIPT_DIR/launch.sh" << 'LAUNCH'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
python "$SCRIPT_DIR/app.py" "$@"
LAUNCH
chmod +x "$SCRIPT_DIR/launch.sh"

echo "    launch.sh created."

# ── Optional model download ───────────────────────────────────────────────────
if [ "$DOWNLOAD_MODELS" -eq 1 ]; then
    echo ""
    echo ">>> Downloading models (this may take a while) …"
    python "$SCRIPT_DIR/download_models.py" --tts 1B --whisper turbo
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  Installation complete!"
echo ""
echo "  To start the UI:"
echo "    bash launch.sh"
echo ""
echo "  To download models separately:"
echo "    source venv/bin/activate"
echo "    python download_models.py --tts 1B --whisper turbo"
echo ""
echo "  To download all models (large download):"
echo "    python download_models.py --all"
echo "======================================================================"
