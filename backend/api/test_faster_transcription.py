import argparse
import time
import logging
import pyaudio
from api.faster_transcription import FasterTranscriber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def print_result(text, result):
    """Callback function to print transcription results."""
    if text:
        print(f"\nTranscription: {text}")
        
        if 'segments' in result and result['segments']:
            print("Segments:")
            for segment in result['segments']:
                print(f"  [{segment['start']:.2f}s -> {segment['end']:.2f}s]: {segment['text']}")

def main():
    parser = argparse.ArgumentParser(description="Test faster-whisper real-time transcription")
    parser.add_argument("--device-name", help="Audio device name (substring)")
    parser.add_argument("--device-index", type=int, help="Audio device index")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    parser.add_argument("--model", default="base", help="Model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--language", help="Language code (e.g., 'en', 'fr') or None for auto-detection")
    parser.add_argument("--compute-type", default="int8", help="Compute type for faster-whisper (int8, float16, etc.)")
    parser.add_argument("--duration", type=int, default=30, help="Duration to run the test in seconds")
    parser.add_argument("--system-audio", action="store_true", help="Try to find and use a virtual audio device")
    args = parser.parse_args()
    
    # Create a temporary transcriber just to list devices
    temp_transcriber = FasterTranscriber()
    
    if args.list_devices:
        devices = temp_transcriber.list_audio_devices()
        print("\nAvailable Audio Input Devices:")
        print("-----------------------------")
        for device in devices:
            virtual_hint = " (Virtual Device)" if device.get('virtual_device_hint', False) else ""
            default_marker = " [DEFAULT]" if device.get('is_default', False) else ""
            print(f"Index {device['index']}: {device['name']}{virtual_hint}{default_marker}")
        return
        
    # Determine device to use
    device = None
    if args.device_index is not None:
        device = args.device_index
    elif args.device_name:
        device = args.device_name
        
    # Configure the transcriber
    transcriber = FasterTranscriber(
        model_name=args.model,
        language=args.language,
        callback=print_result,
        device=device,
        use_system_audio=args.system_audio,
        compute_type=args.compute_type
    )
    
    # Start transcription
    transcriber.start()

    try:
        print(f"\nListening... (press Ctrl+C to stop, running for {args.duration} seconds)")
        print(f"Model: {args.model}, Language: {args.language or 'auto'}, Device: {device or 'default'}")
        start_time = time.time()
        
        # Run for specified duration
        while time.time() - start_time < args.duration:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Stop transcription
        transcriber.stop()
        print("\nTranscription stopped.")

if __name__ == "__main__":
    main()
