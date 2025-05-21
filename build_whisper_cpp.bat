@echo off
setlocal enabledelayedexpansion

REM --- Configuration ---
set WHISPER_CPP_DIR=%~dp0whisper_cpp_installation\whisper.cpp
set MODEL_NAME=ggml-base.en.bin
set MODEL_URL=https://huggingface.co/ggerganov/whisper.cpp/resolve/main/%MODEL_NAME%

echo ============================================================
echo          Whisper.cpp Build Script for Windows
echo ============================================================
echo.

REM --- Check for CMake ---
cmake --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake is not installed or not in PATH.
    echo Please install CMake from https://cmake.org/download/
    echo and ensure it's added to your system PATH.
    goto :error_exit
)
echo [OK] CMake is installed

cd /D "%WHISPER_CPP_DIR%"

REM --- Create build directory ---
if not exist "build" (
    echo [INFO] Creating build directory...
    mkdir build
    if errorlevel 1 (
        echo [ERROR] Failed to create build directory. Please check permissions.
        goto :error_exit
    )
)

REM --- Build with CMake ---
cd build
echo [INFO] Configuring build with CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 (
    echo [ERROR] CMake configuration failed.
    goto :error_exit
)

echo [INFO] Building whisper.cpp with CMake...
cmake --build . --config Release
if errorlevel 1 (
    echo [ERROR] Build failed.
    goto :error_exit
)

REM --- Create models directory if it doesn't exist ---
cd ..
if not exist "models" mkdir models

REM --- Download model if not present ---
if not exist "models\%MODEL_NAME%" (
    echo [INFO] Downloading model %MODEL_NAME%...
    echo This may take several minutes depending on your internet connection...
    
    powershell -Command "if (!(Test-Path -Path 'models\%MODEL_NAME%')) { Invoke-WebRequest -Uri '%MODEL_URL%' -OutFile 'models\%MODEL_NAME%' }"
    if errorlevel 1 (
        echo [WARNING] PowerShell download failed, trying with curl if available...
        curl -L -o "models\%MODEL_NAME%" "%MODEL_URL%"
        if errorlevel 1 (
            echo [ERROR] Failed to download model.
            echo You can manually download it from:
            echo %MODEL_URL%
            echo And place it in: %WHISPER_CPP_DIR%\models\
            goto :error_exit
        )
    )
    echo [OK] Model downloaded successfully
) else (
    echo [INFO] Model %MODEL_NAME% already exists
)

REM --- Check for executables ---
set MAIN_EXE=
set STREAM_EXE=

if exist "build\bin\Release\main.exe" (
    set "MAIN_EXE=build\bin\Release\main.exe"
) else if exist "build\main.exe" (
    set "MAIN_EXE=build\main.exe"
)

if exist "build\bin\Release\stream.exe" (
    set "STREAM_EXE=build\bin\Release\stream.exe"
) else if exist "build\stream.exe" (
    set "STREAM_EXE=build\stream.exe"
)

if not defined MAIN_EXE (
    echo [ERROR] Could not find main.exe after compilation.
    goto :error_exit
)

if not defined STREAM_EXE (
    echo [ERROR] Could not find stream.exe after compilation.
    goto :error_exit
)

echo.
echo ============================================================
echo                  BUILD COMPLETED SUCCESSFULLY!
echo ============================================================
echo.
echo [SUCCESS] whisper.cpp is ready to use!
echo.
echo Main Executable: %CD%\%MAIN_EXE%
echo Stream Executable: %CD%\%STREAM_EXE%
echo Model: %CD%\models\%MODEL_NAME%
echo.
echo These paths have been updated in the backend configuration.
echo.
pause
goto :eof

:error_exit
echo.
echo ============================================================
echo                  BUILD FAILED WITH ERRORS
echo ============================================================
echo.
echo Please check the error messages above for details.
echo.
pause
exit /b 1

endlocal
