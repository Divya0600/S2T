import os
import queue
import threading
import time
import logging
import numpy as np
import pyaudio
import whisper
import signal
import re
from typing import Optional, Dict, Any, List, Callable, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class RealtimeTranscriber:
    """
    A class to handle real-time audio transcription using the Whisper model.
    """
    def __init__(
        self,
        model_name: str = "base",
        language: Optional[str] = None,
        chunk_duration_ms: int = 5000,
        sample_rate: int = 16000,
        channels: int = 1,
        format_type: int = pyaudio.paInt16,
        device: Optional[Union[int, str]] = None,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        use_system_audio: bool = False
    ):
        """
        Initialize the real-time transcriber.
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            language: Language code (e.g., 'en', 'fr') or None for auto-detection
            chunk_duration_ms: Duration of each audio chunk in milliseconds
            sample_rate: Audio sample rate (Whisper expects 16000)
            channels: Number of audio channels (1 for mono)
            format_type: PyAudio format type
            device_index: Index of the input device to use, or None for default
            callback: Optional callback function for receiving transcription results
        """
        self.model_name = model_name
        self.language = language
        self.chunk_duration_ms = chunk_duration_ms
        self.sample_rate = sample_rate
        self.channels = channels
        self.format_type = format_type
        self.device = device  # Can be index (int) or name (str)
        self.callback = callback
        self.use_system_audio = use_system_audio
        
        # Will be set during initialization
        self.device_index = None
        
        # Audio parameters
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self.chunk_bytes = self.chunk_samples * channels * 2  # 2 bytes per sample for paInt16
        
        # State flags
        self.running = False
        self.paused = False
        
        # Threading and queue for audio processing
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.lock = threading.Lock()
        
        # Load the Whisper model
        logger.info(f"Loading Whisper model: {model_name}")
        try:
            self.model = whisper.load_model(model_name)
            logger.info(f"Whisper model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, sig, frame):
        """Handle termination signals for graceful shutdown."""
        logger.info(f"Received signal {sig}, stopping transcription...")
        self.stop()
    
    def _find_device_index(self, device_identifier: Optional[Union[int, str]] = None) -> Optional[int]:
        """
        Find the device index based on the provided identifier.
        
        Args:
            device_identifier: Either a device index (int) or a device name/pattern (str)
                              If None, will try to find a suitable default or virtual device
        
        Returns:
            The device index or None if not found
        """
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        
        # List of patterns that might indicate virtual audio devices
        virtual_device_patterns = [
            r"vb-audio", r"blackhole", r"virtual", r"cable", r"stereo mix",
            r"loopback", r"wasapi", r"monitor", r"soundflower"
        ]
        
        try:
            # Case 1: No device specified, use system default
            if device_identifier is None:
                # First try to find a virtual audio device if use_system_audio is True
                if self.use_system_audio:
                    for i in range(device_count):
                        info = p.get_device_info_by_index(i)
                        if info['maxInputChannels'] > 0:  # Input device
                            for pattern in virtual_device_patterns:
                                if re.search(pattern, info['name'].lower()):
                                    logger.info(f"Found virtual audio device: {info['name']} (index: {i})")
                                    return i
                
                # If no virtual device found or not looking for one, use default
                default_index = p.get_default_input_device_info()['index']
                logger.info(f"Using default audio device (index: {default_index})")
                return default_index
            
            # Case 2: Device index provided
            elif isinstance(device_identifier, int):
                if 0 <= device_identifier < device_count:
                    info = p.get_device_info_by_index(device_identifier)
                    if info['maxInputChannels'] > 0:  # Ensure it's an input device
                        logger.info(f"Using specified audio device index: {device_identifier}")
                        return device_identifier
                    else:
                        logger.warning(f"Device index {device_identifier} is not an input device")
                else:
                    logger.warning(f"Device index {device_identifier} out of range (0-{device_count-1})")
            
            # Case 3: Device name or pattern provided
            elif isinstance(device_identifier, str):
                # First try exact match
                for i in range(device_count):
                    info = p.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0 and device_identifier.lower() == info['name'].lower():
                        logger.info(f"Found exact match for device name: {info['name']} (index: {i})")
                        return i
                
                # Then try partial/case-insensitive match
                for i in range(device_count):
                    info = p.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0 and device_identifier.lower() in info['name'].lower():
                        logger.info(f"Found partial match for device name: {info['name']} (index: {i})")
                        return i
                
                logger.warning(f"No device found matching name: {device_identifier}")
            
            # Default behavior if we couldn't find a matching device
            if p.get_default_input_device_info():
                default_index = p.get_default_input_device_info()['index']
                logger.warning(f"Using default input device as fallback (index: {default_index})")
                return default_index
            
            return None
        
        except Exception as e:
            logger.error(f"Error finding audio device: {e}")
            return None
        
        finally:
            p.terminate()

    def list_audio_devices(self, include_virtual_hint: bool = True) -> List[Dict[str, Any]]:
        """
        List all available audio input devices.
        
        Args:
            include_virtual_hint: Whether to add a hint if a device appears to be a virtual audio device
            
        Returns:
            A list of dictionaries with device information
        """
        p = pyaudio.PyAudio()
        devices = []
        
        # Patterns that might indicate virtual audio devices
        virtual_device_patterns = [
            r"vb-audio", r"blackhole", r"virtual", r"cable", r"stereo mix",
            r"loopback", r"wasapi", r"monitor", r"soundflower"
        ]
        
        try:
            for i in range(p.get_device_count()):
                try:
                    device_info = p.get_device_info_by_index(i)
                    if device_info['maxInputChannels'] > 0:  # Only input devices
                        device_data = {
                            'index': i,
                            'name': device_info['name'],
                            'channels': device_info['maxInputChannels'],
                            'sample_rate': int(device_info['defaultSampleRate'])
                        }
                        
                        # Check if this might be a virtual audio device
                        if include_virtual_hint:
                            for pattern in virtual_device_patterns:
                                if re.search(pattern, device_info['name'].lower()):
                                    device_data['virtual_device_hint'] = True
                                    break
                        
                        devices.append(device_data)
                except Exception as e:
                    logger.warning(f"Error getting device info for index {i}: {e}")
        finally:
            p.terminate()
            
        return devices
    
    def start(self):
        """Start audio capture and transcription."""
        if self.running:
            logger.warning("Transcription is already running")
            return
        
        logger.info("Starting real-time transcription")
        self.running = True
        self.paused = False
        
        # Start the processing thread
        self.processing_thread = threading.Thread(
            target=self._process_audio_thread,
            daemon=True
        )
        self.processing_thread.start()
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Determine the device index based on the provided device identifier
        self.device_index = self._find_device_index(self.device)
        if self.device_index is None:
            logger.error("No suitable audio input device found")
            self.running = False
            raise ValueError("No suitable audio input device found")
        
        # Get device info for logging
        device_info = self.audio.get_device_info_by_index(self.device_index)
        logger.info(f"Using audio device: {device_info['name']} (index: {self.device_index})")
            
        # Start audio stream
        try:
            self.stream = self.audio.open(
                format=self.format_type,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_samples,
                stream_callback=self._audio_callback
            )
            logger.info("Audio stream started")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self.running = False
            self.audio.terminate()
            raise
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for the PyAudio stream."""
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        if not self.paused and self.running:
            self.audio_queue.put(in_data)
        
        return (None, pyaudio.paContinue)
    
    def _process_audio_thread(self):
        """Thread for processing audio chunks and running transcription."""
        logger.info("Audio processing thread started")
        
        buffer = bytearray()
        while self.running:
            try:
                # Get audio chunk from queue with timeout
                try:
                    chunk = self.audio_queue.get(timeout=0.5)
                    buffer.extend(chunk)
                except queue.Empty:
                    continue
                
                # Check if we have enough audio to process
                if len(buffer) >= self.chunk_bytes:
                    # Convert buffer to numpy array for Whisper
                    audio_data = np.frombuffer(buffer[:self.chunk_bytes], np.int16).astype(np.float32) / 32768.0
                    
                    # Transcribe the audio chunk
                    result = self._transcribe_audio(audio_data)
                    
                    # Send result to callback if provided
                    if self.callback and result:
                        self.callback(result['text'], result)
                    
                    # Also store in result queue
                    self.result_queue.put(result)
                    
                    # Reset buffer with any remaining audio
                    buffer = buffer[self.chunk_bytes:]
            
            except Exception as e:
                logger.error(f"Error in audio processing thread: {e}")
                time.sleep(0.1)  # Prevent tight loop on error
        
        logger.info("Audio processing thread stopped")
    
    def _transcribe_audio(self, audio_data):
        """
        Transcribe an audio chunk using the Whisper model.
        
        Args:
            audio_data: NumPy array of audio samples
            
        Returns:
            Transcription result dictionary
        """
        try:
            # Transcribe with Whisper
            options = {
                "language": self.language,
                "task": "transcribe"
            }
            
            # Skip if audio is too quiet
            if np.abs(audio_data).max() < 0.01:
                return {"text": ""}
            
            result = self.model.transcribe(
                audio_data,
                **{k: v for k, v in options.items() if v is not None}
            )
            
            return result
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"text": "", "error": str(e)}
    
    def pause(self):
        """Pause transcription without stopping the audio stream."""
        logger.info("Pausing transcription")
        self.paused = True
    
    def resume(self):
        """Resume transcription after pausing."""
        logger.info("Resuming transcription")
        self.paused = False
    
    def stop(self):
        """Stop audio capture and transcription."""
        if not self.running:
            logger.warning("Transcription is not running")
            return
        
        logger.info("Stopping transcription")
        
        # Set flag to stop threads
        self.running = False
        
        # Wait for a moment to allow threads to clean up
        time.sleep(0.5)
        
        # Close and clean up audio resources
        try:
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            if hasattr(self, 'audio') and self.audio:
                self.audio.terminate()
                
            logger.info("Audio resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up audio resources: {e}")
        
        # Wait for processing thread to complete
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
            logger.info("Processing thread joined")
    
    def get_latest_result(self, block=False, timeout=None):
        """
        Get the latest transcription result from the queue.
        
        Args:
            block: Whether to block until a result is available
            timeout: Timeout in seconds if blocking
            
        Returns:
            The latest transcription result or None
        """
        try:
            return self.result_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def clear_queues(self):
        """Clear all pending audio and result queues."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Queues cleared")

# Singleton instance for application-wide use
_transcriber_instance = None

def get_transcriber(config=None):
    """
    Get or create the global transcriber instance.
    
    Args:
        config: Optional configuration dictionary to initialize the transcriber
        
    Returns:
        The global RealtimeTranscriber instance
    """
    global _transcriber_instance
    
    if _transcriber_instance is None and config is not None:
        # Handle device parameter, which can be index (int) or name (str)
        if 'device_index' in config and 'device' not in config:
            # For backward compatibility
            config['device'] = config.pop('device_index')
            
        _transcriber_instance = RealtimeTranscriber(**config)
    
    return _transcriber_instance
