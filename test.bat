@echo off
setlocal EnableDelayedExpansion

echo ---------------------------------------
echo 🔍 Checking for FFmpeg...
echo ---------------------------------------

where ffmpeg >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ FFmpeg not found in PATH.
    echo ➤ Please install it from https://www.gyan.dev/ffmpeg/builds/
    echo ➤ Then add the /bin folder to your System PATH.
    pause
    exit /b 1
) else (
    echo ✅ FFmpeg is installed!
)

echo.
echo ---------------------------------------
echo 📦 Installing Python dependencies...
echo ---------------------------------------
pip install --upgrade pip
pip install openai-whisper torch >nul 2>&1

echo ✅ Dependencies installed.

echo.
echo ---------------------------------------
echo 💾 Writing transcription script to live_transcribe.py...
echo ---------------------------------------

REM Write the Python script
(
echo import whisper
echo import subprocess
echo import time
echo import os
echo.
echo model = whisper.load_model("tiny")  ^# or 'base'
echo.
echo record_cmd = [
echo     "ffmpeg",
echo     "-f", "dshow",
echo     "-i", "audio=default",
echo     "-t", "5",
echo     "-y", "live.wav"
echo ]
echo.
echo print("🎙️ Live transcription starting (default audio input)...")
echo print("⏱️ Ctrl+C to stop.\n")
echo.
echo try:
echo     while True:
echo         subprocess.run(record_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
echo         result = model.transcribe("live.wav", fp16=False, language="en")
echo         print(">>", result["text"].strip())
echo.
echo except KeyboardInterrupt:
echo     print("\n🛑 Transcription stopped.")
echo     if os.path.exists("live.wav"):
echo         os.remove("live.wav")
) > live_transcribe.py

echo ✅ Script created: live_transcribe.py

echo.
echo ---------------------------------------
choice /M "🚀 Do you want to run the transcription script now?"
if %ERRORLEVEL%==1 (
    python live_transcribe.py
) else (
    echo 👌 You can run it anytime with: python live_transcribe.py
)

exit /b
