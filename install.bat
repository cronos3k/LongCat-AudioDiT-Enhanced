@echo off
REM =============================================================================
REM  install.bat — LongCat-AudioDiT Enhanced installer for Windows
REM =============================================================================
REM  Usage:
REM    install.bat                    :: CUDA build (default)
REM    install.bat --cpu              :: CPU-only build
REM    install.bat --download-models  :: download models after install
REM =============================================================================

setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%venv
set REQ_FILE=%SCRIPT_DIR%requirements_enhanced.txt
set CPU_ONLY=0
set DOWNLOAD_MODELS=0
set PYTHON_BIN=python

REM ── Parse arguments ──────────────────────────────────────────────────────
for %%a in (%*) do (
    if "%%a"=="--cpu"             set CPU_ONLY=1
    if "%%a"=="--download-models" set DOWNLOAD_MODELS=1
)

echo ======================================================================
echo   LongCat-AudioDiT Enhanced ^— Windows Installer
echo ======================================================================
echo   Script dir : %SCRIPT_DIR%
echo   Venv dir   : %VENV_DIR%
echo   CPU only   : %CPU_ONLY%
echo ======================================================================

REM ── Check Python ─────────────────────────────────────────────────────────
echo.
echo ^>^>^> Checking Python version ...
%PYTHON_BIN% -c "import sys; v=sys.version_info; print(f'Python {v.major}.{v.minor}.{v.micro}'); sys.exit(0 if (v.major,v.minor)>=(3,10) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.10 or later is required.
    echo Download from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM ── Create virtual environment ───────────────────────────────────────────
echo.
echo ^>^>^> Creating virtual environment at %VENV_DIR% ...
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo     Venv already exists -- reusing.
) else (
    %PYTHON_BIN% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
echo     Activated: %VIRTUAL_ENV%

REM ── Upgrade pip ──────────────────────────────────────────────────────────
echo.
echo ^>^>^> Upgrading pip ...
pip install --upgrade pip setuptools wheel -q

REM ── Install PyTorch ──────────────────────────────────────────────────────
echo.
echo ^>^>^> Installing PyTorch ...
if "%CPU_ONLY%"=="1" (
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
) else (
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo     CUDA 12.4 failed, trying CUDA 12.1 ...
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    )
)

REM ── Install project requirements ─────────────────────────────────────────
echo.
echo ^>^>^> Installing project requirements ...
pip install -r "%REQ_FILE%"
if errorlevel 1 (
    echo ERROR: Requirements installation failed.
    pause
    exit /b 1
)

REM ── Verify imports ───────────────────────────────────────────────────────
echo.
echo ^>^>^> Verifying installation ...
python -c "import torch, torchaudio, transformers, gradio, faster_whisper, soundfile, librosa; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('All imports OK.')"
if errorlevel 1 (
    echo WARNING: Some imports failed. Check the output above.
)

REM ── Write launch script ───────────────────────────────────────────────────
echo.
echo ^>^>^> Writing launch.bat ...
(
    echo @echo off
    echo call "%VENV_DIR%\Scripts\activate.bat"
    echo python "%SCRIPT_DIR%app.py" %%*
) > "%SCRIPT_DIR%launch.bat"
echo     launch.bat created.

REM ── Optional model download ───────────────────────────────────────────────
if "%DOWNLOAD_MODELS%"=="1" (
    echo.
    echo ^>^>^> Downloading models ...
    python "%SCRIPT_DIR%download_models.py" --tts 1B --whisper turbo
)

REM ── Done ─────────────────────────────────────────────────────────────────
echo.
echo ======================================================================
echo   Installation complete!
echo.
echo   To start the UI:
echo     launch.bat
echo.
echo   To download models separately:
echo     venv\Scripts\activate.bat
echo     python download_models.py --tts 1B --whisper turbo
echo.
echo   To download all models (large download):
echo     python download_models.py --all
echo ======================================================================
pause
