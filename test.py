import subprocess
import os
import time
from faster_whisper import WhisperModel

# Load the model
model = WhisperModel("tiny", compute_type="int8")

# Device selection with fallback options
mics = [
    "Headset Microphone (Plantronics Blackwire 3220 Series)",
    "Microphone Array (Realtek(R) Audio)",
    "default"
]

def try_recording(mic_name):
    cmd = [
        "ffmpeg",
        "-f", "dshow",
        "-i", f"audio={mic_name}",
        "-t", "5",
        "-y", "live.wav"
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=7)
        return os.path.exists("live.wav")
    except:
        return False

print("🎤 Live transcription with Faster-Whisper")
print("Testing microphones...")

# Try each mic until one works
working_mic = None
for mic in mics:
    print(f"Testing: {mic}")
    if try_recording(mic):
        working_mic = mic
        print(f"Using: {mic}")
        break

if not working_mic:
    print("No working microphone found. Please check your audio settings.")
    exit(1)

input("Press Enter to start live transcription... ")
print("⏱️ Speaking now... Ctrl+C to stop\n")

record_cmd = [
    "ffmpeg",
    "-f", "dshow",
    "-i", f"audio={working_mic}",
    "-t", "5",
    "-y", "live.wav"
]

try:
    while True:
        subprocess.run(record_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists("live.wav") and os.path.getsize("live.wav") > 1000:
            segments, _ = model.transcribe("live.wav", language="en")
            text = " ".join(segment.text.strip() for segment in segments)
            print(">>", text if text.strip() else "(No speech detected)")
        else:
            print("Recording failed, retrying...")
            time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Transcription stopped.")
    if os.path.exists("live.wav"):
        os.remove("live.wav")