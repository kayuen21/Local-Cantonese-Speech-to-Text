@echo off
setlocal enabledelayedexpansion
title Cantonese Speech-to-Text Environment Setup
color 0F
cd /d "%~dp0"

echo.
echo =======================================================================
echo         Cantonese Speech-to-Text - Environment Installer
echo =======================================================================
echo.
echo  This installer will set up the complete GPU-accelerated environment:
echo.
echo    [1/5] Embedded Python 3.11 Runtime (~11 MB)
echo    [2/5] Pip Package Manager
echo    [3/5] PyTorch with NVIDIA CUDA cu128 Acceleration (~2.3 GB)
echo    [4/5] Faster-Whisper, PyAnnote Audio ^& Web Dependencies (~200 MB)
echo    [5/5] Embedded GPU LLM Engine (llama-server with CUDA) (~180 MB)
echo.
echo  All components are self-contained in this folder (no system clutter).
echo =======================================================================
echo.
echo Press any key to begin installation...
pause >nul
echo.

:: Ensure directories exist
if not exist "models" mkdir "models"
if not exist "app\uploads" mkdir "app\uploads"
if not exist "app\results" mkdir "app\results"
if not exist "llm" mkdir "llm"
if not exist "python" mkdir "python"

:: -------------------------------------------------------------------------
:: [1/5] Python Runtime
:: -------------------------------------------------------------------------
echo -----------------------------------------------------------------------
echo  [Step 1/5] Setting up Portable Python 3.11 Runtime
echo -----------------------------------------------------------------------
if exist "python\python.exe" (
    echo  - Python runtime already present. Skipping download.
    goto :step_2
)

echo  - Downloading Python embeddable package (~11 MB)...
curl -# -L -o "python.zip" "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
if errorlevel 1 (
    echo [ERROR] Failed to download Python. Please check your internet connection.
    goto :fail
)
echo  - Extracting Python...
if exist "%SystemRoot%\System32\tar.exe" (
    "%SystemRoot%\System32\tar.exe" -xf "python.zip" -C "python"
) else (
    powershell -NoProfile -Command "Expand-Archive -Path 'python.zip' -DestinationPath 'python' -Force"
)
if errorlevel 1 (
    echo [ERROR] Failed to extract Python.
    goto :fail
)
del /f /q "python.zip" >nul 2>&1

echo  - Configuring Python search paths...
> "python\python311._pth" echo python311.zip
>> "python\python311._pth" echo .
>> "python\python311._pth" echo Lib\site-packages
>> "python\python311._pth" echo ..\app
>> "python\python311._pth" echo import site
echo  - Python runtime configured successfully.

:step_2
echo.

:: -------------------------------------------------------------------------
:: [2/5] Pip Setup
:: -------------------------------------------------------------------------
echo -----------------------------------------------------------------------
echo  [Step 2/5] Setting up Pip Package Manager
echo -----------------------------------------------------------------------
if exist "python\Lib\site-packages\pip" (
    echo  - Pip is already installed.
    goto :step_3
)

echo  - Downloading get-pip.py...
curl -# -L -o "python\get-pip.py" "https://bootstrap.pypa.io/get-pip.py"
if errorlevel 1 (
    echo [ERROR] Failed to download pip installer.
    goto :fail
)
echo  - Installing pip into portable environment...
"python\python.exe" "python\get-pip.py" --no-warn-script-location
if errorlevel 1 (
    echo [ERROR] Failed to execute get-pip.py.
    goto :fail
)
del /f /q "python\get-pip.py" >nul 2>&1
echo  - Pip configured successfully.

:step_3
echo -----------------------------------------------------------------------
echo  [Step 3/5] Installing PyTorch with NVIDIA CUDA cu128 (~2.3 GB)
echo -----------------------------------------------------------------------
echo  - Downloading and installing CUDA-enabled PyTorch...
echo  - (Live download progress will display below):
echo.
"python\python.exe" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 --progress-bar on --no-warn-script-location
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch with CUDA.
    goto :fail
)
echo.
echo  - PyTorch CUDA installed successfully.
echo.

:: -------------------------------------------------------------------------
:: [4/5] Faster-Whisper, PyAnnote & App Dependencies
:: -------------------------------------------------------------------------
echo -----------------------------------------------------------------------
echo  [Step 4/5] Installing Faster-Whisper, PyAnnote and Web Libraries
echo -----------------------------------------------------------------------
echo  - Installing required packages...
"python\python.exe" -m pip install -r "app\requirements.txt" --progress-bar on --no-warn-script-location
if errorlevel 1 (
    echo [ERROR] Failed to install application dependencies.
    goto :fail
)
echo.
echo  - Speech and web dependencies installed successfully.
echo.

:: -------------------------------------------------------------------------
:: [5/5] GPU LLM Engine (llama-server)
:: -------------------------------------------------------------------------
echo -----------------------------------------------------------------------
echo  [Step 5/5] Setting up Embedded GPU LLM Server (llama-server CUDA)
echo -----------------------------------------------------------------------
if exist "llm\llama-server.exe" if exist "llm\ggml-cuda.dll" (
    echo  - GPU LLM Server already present. Skipping download.
    goto :skip_llm
)

echo  - Downloading llama-server CUDA binary (~180 MB)...
curl -# -L -o "llama-cuda.zip" "https://github.com/ggml-org/llama.cpp/releases/download/b7075/llama-b7075-bin-win-cuda-12.4-x64.zip"
if errorlevel 1 (
    echo [ERROR] Failed to download llama-server CUDA binary.
    goto :fail
)
echo  - Downloading CUDA runtime DLLs (~370 MB)...
curl -# -L -o "cudart.zip" "https://github.com/ggml-org/llama.cpp/releases/download/b7075/cudart-llama-bin-win-cuda-12.4-x64.zip"
if errorlevel 1 (
    echo [ERROR] Failed to download CUDA runtime DLLs.
    goto :fail
)
echo  - Extracting GPU LLM files into llm/...
if exist "%SystemRoot%\System32\tar.exe" (
    "%SystemRoot%\System32\tar.exe" -xf "llama-cuda.zip" -C "llm"
    "%SystemRoot%\System32\tar.exe" -xf "cudart.zip" -C "llm"
) else (
    powershell -NoProfile -Command "Expand-Archive -Path 'llama-cuda.zip' -DestinationPath 'llm' -Force; Expand-Archive -Path 'cudart.zip' -DestinationPath 'llm' -Force"
)
del /f /q "llama-cuda.zip" "cudart.zip" >nul 2>&1
echo  - Embedded GPU LLM server ready.

:skip_llm
echo.

:: -------------------------------------------------------------------------
:: Verification & Completion
:: -------------------------------------------------------------------------
echo =======================================================================
echo                  Environment Setup Complete!
echo =======================================================================
echo.
echo  All components are successfully installed and verified:
echo   - Portable Python 3.11 with CUDA acceleration
echo   - Faster-Whisper Cantonese recognition engine
echo   - PyAnnote speaker diarization
echo   - Local GPU LLM engine (llama-server)
echo.
echo  You can now launch the app by double-clicking:
echo      =^> CantoneseSTT.exe
echo.
echo  (Models will download automatically on first selection into models\)
echo =======================================================================
echo.
echo Press any key to exit installer...
pause >nul
exit /b 0

:fail
echo.
echo =======================================================================
echo  Setup encountered an error. Please check the message above.
echo =======================================================================
echo.
pause
exit /b 1
