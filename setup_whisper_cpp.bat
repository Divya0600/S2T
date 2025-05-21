@echo off
setlocal enabledelayedexpansion

REM --- Configuration ---
set WHISPER_CPP_DIR=%~dp0whisper_cpp_installation
set MODEL_NAME=ggml-base.en.bin
set MODEL_URL=https://huggingface.co/ggerganov/whisper.cpp/resolve/main/%MODEL_NAME%
set WHISPER_REPO_URL=https://github.com/ggerganov/whisper.cpp.git

REM --- Check if whisper.cpp is already installed and working ---
set WHISPER_READY=0
if exist "%WHISPER_CPP_DIR%\whisper.cpp\stream.exe" (
    echo [INFO] Found existing whisper.cpp stream executable
    set "STREAM_EXE=%WHISPER_CPP_DIR%\whisper.cpp\stream.exe"
    set WHISPER_READY=1
) else if exist "%WHISPER_CPP_DIR%\whisper.cpp\build\bin\Release\stream.exe" (
    echo [INFO] Found existing whisper.cpp stream executable (CMake build)
    set "STREAM_EXE=%WHISPER_CPP_DIR%\whisper.cpp\build\bin\Release\stream.exe"
    set WHISPER_READY=1
)

if %WHISPER_READY% == 1 (
    if exist "%WHISPER_CPP_DIR%\whisper.cpp\models\%MODEL_NAME%" (
        echo.
        echo ============================================================
        echo  whisper.cpp is already installed and ready to use!
        echo  Stream Executable: %STREAM_EXE%
        echo  Model: %WHISPER_CPP_DIR%\whisper.cpp\models\%MODEL_NAME%
        echo.
        echo  Would you like to reinstall? (Y/N)
        set /p REINSTALL=
        if /i not "!REINSTALL!"=="Y" (
            echo Installation skipped. Using existing installation.
            goto :end_install
        )
    )
)

echo ============================================================
echo            Whisper.cpp Setup Script for Windows
echo ============================================================
echo.
echo This script will:
echo 1. Check for required tools (Git, C++ compiler)
echo 2. Create directory: %WHISPER_CPP_DIR%
echo 3. Clone/Update whisper.cpp repository
if not exist "%WHISPER_CPP_DIR%\whisper.cpp\models\%MODEL_NAME%" (
    echo 4. Download the '%MODEL_NAME%' model
) else (
    echo 4. Model '%MODEL_NAME%' already exists
)
echo 5. Compile whisper.cpp (if needed)
echo.
pause

REM --- Check for Git ---
git --version >nul 2>&1
if errorlevel 9009 (
    echo [ERROR] Git is not installed or not in PATH.
    echo Please install Git from https://git-scm.com/download/win
    echo and ensure it's added to your system PATH.
    goto :error_exit
)
echo [OK] Git is installed

REM --- Create directory ---
if not exist "%WHISPER_CPP_DIR%" (
    echo [INFO] Creating directory: %WHISPER_CPP_DIR%
    mkdir "%WHISPER_CPP_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create directory. Please check permissions.
        goto :error_exit
    )
) else (
    echo [INFO] Directory %WHISPER_CPP_DIR% already exists
)

cd /D "%WHISPER_CPP_DIR%"

REM --- Clone/Update whisper.cpp repository ---
if not exist "whisper.cpp\.git" (
    echo [INFO] Cloning whisper.cpp repository...
    git clone %WHISPER_REPO_URL%
    if errorlevel 1 (
        echo [ERROR] Failed to clone whisper.cpp repository.
        goto :error_exit
    )
    echo [OK] Repository cloned successfully
) else (
    echo [INFO] Updating whisper.cpp repository...
    cd whisper.cpp
    git fetch
    git reset --hard origin/master
    cd ..
    echo [OK] Repository updated
)

cd whisper.cpp

REM --- Create models directory if it doesn't exist ---
if not exist "models" mkdir models

REM --- Download model if not present ---
if not exist "models\%MODEL_NAME%" (
    echo [INFO] Downloading model %MODEL_NAME%...
    echo This may take several minutes depending on your internet connection...
    
    REM Try with PowerShell first, fall back to curl if available
    powershell -Command "if (!(Test-Path -Path 'models\%MODEL_NAME%')) { Invoke-WebRequest -Uri '%MODEL_URL%' -OutFile 'models\%MODEL_NAME%' }"
    if errorlevel 1 (
        echo [WARNING] PowerShell download failed, trying with curl if available...
        curl -L -o "models\%MODEL_NAME%" "%MODEL_URL%"
        if errorlevel 1 (
            echo [ERROR] Failed to download model.
            echo You can manually download it from:
            echo %MODEL_URL%
            echo And place it in: %WHISPER_CPP_DIR%\whisper.cpp\models\
            echo Then run this script again.
            goto :error_exit
        )
    )
    echo [OK] Model downloaded successfully
) else (
    echo [INFO] Model %MODEL_NAME% already exists
)

REM --- Check if compilation is needed ---
set COMPILE_NEEDED=1
if exist "stream.exe" (
    echo [INFO] Found existing stream.exe
    set COMPILE_NEEDED=0
) else if exist "build\bin\Release\stream.exe" (
    echo [INFO] Found existing stream.exe (CMake build)
    set COMPILE_NEEDED=0
)

if %COMPILE_NEEDED% == 1 (
    echo [INFO] Compiling whisper.cpp...
    
    REM Try with make first
    echo [INFO] Trying to compile with 'make'...
    make stream
    if errorlevel 1 (
        echo [WARNING] 'make' failed or not available, trying CMake...
        
        REM Try with CMake
        if not exist "build" mkdir build
        cd build
        cmake ..
        if errorlevel 1 (
            echo [ERROR] CMake configuration failed.
            echo Please ensure you have CMake and a C++ compiler installed.
            goto :error_exit
        )
        cmake --build . --config Release
        if errorlevel 1 (
            echo [ERROR] Build failed.
            goto :error_exit
        )
        cd ..
    )
    echo [OK] Compilation completed successfully
) else (
    echo [INFO] Compilation not needed - existing binaries found
)

REM --- Verify installation ---
set STREAM_EXE=
if exist "stream.exe" set "STREAM_EXE=%CD%\stream.exe"
if exist "build\bin\Release\stream.exe" set "STREAM_EXE=%CD%\build\bin\Release\stream.exe"

if not defined STREAM_EXE (
    echo [ERROR] Could not find stream executable after compilation.
    goto :error_exit
)

:end_install
echo.
echo ============================================================
echo                  SETUP COMPLETED SUCCESSFULLY!
echo ============================================================
echo.
echo [SUCCESS] whisper.cpp is ready to use!
echo.
echo Stream Executable: %STREAM_EXE%
echo Model: %WHISPER_CPP_DIR%\whisper.cpp\models\%MODEL_NAME%
echo.
echo To test the installation, you can run:
echo   cd /d "%WHISPER_CPP_DIR%\whisper.cpp"
echo   %STREAM_EXE% -m models\%MODEL_NAME% -t 8 --step 0 --length 5000 -v
echo.
echo For live transcription integration, use this path:
echo %STREAM_EXE%
echo.
pause
goto :eof

:error_exit
echo.
echo ============================================================
echo                  SETUP FAILED WITH ERRORS
echo ============================================================
echo.
echo Please check the error messages above for details.
echo Common issues:
echo 1. Make sure Git is installed and in your PATH
if not exist "%WHISPER_CPP_DIR%\whisper.cpp\models\%MODEL_NAME%" (
echo 2. You can manually download the model from:
echo    %MODEL_URL%
echo    and place it in: %WHISPER_CPP_DIR%\whisper.cpp\models\
)
echo 3. Ensure you have a C++ compiler installed:
echo    - For MinGW-w64: Install from https://www.mingw-w64.org/
echo    - For Visual Studio: Install 'Desktop development with C++' workload
echo.
pause
exit /b 1

endlocal
