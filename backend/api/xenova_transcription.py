"""
Xenova Transcription Engine - Alternative to faster_whisper for live transcription
Uses the Xenova implementation for transcription which can be more stable in some environments
"""

import os
import time
import queue
import logging
import tempfile
import threading
import subprocess
import numpy as np
from typing import Dict, Any, Optional, List, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global instance for the transcriber
_xenova_transcriber_instance = None

def get_xenova_transcriber(config=None):
    """Get the global Xenova transcriber instance, creating it if needed."""
    global _xenova_transcriber_instance
    if _xenova_transcriber_instance is None and config is not None:
        _xenova_transcriber_instance = XenovaTranscriber(**config)
    return _xenova_transcriber_instance

class XenovaTranscriber:
    """
    Transcriber using Xenova's implementation of Whisper.
    This serves as an alternative to the FasterTranscriber in case of compatibility issues.
    """
    
    def __init__(
        self,
        model_name: str = "base",
        language: Optional[str] = None,
        device: str = "cpu",
        chunk_duration_ms: int = 5000,
        use_system_audio: bool = False,
        **kwargs
    ):
        """
        Initialize the Xenova transcriber with the given parameters.
        
        Args:
            model_name: The Whisper model to use (tiny, base, small, medium, large)
            language: The language code, or None for auto-detection
            device: The device to use ("cpu" or "cuda")
            chunk_duration_ms: Duration of audio chunks for processing
            use_system_audio: Whether to capture system audio
        """
        self.model_name = model_name
        self.language = language
        self.device = device
        self.chunk_duration_ms = chunk_duration_ms
        self.use_system_audio = use_system_audio
        
        # State management
        self.is_running = False
        self.results_queue = queue.Queue()
        self.latest_result = {"text": "", "segments": []}
        
        # The transcription thread
        self.transcription_thread = None
        
        logger.info(f"Initialized Xenova transcriber with model: {model_name}")
    
    def start(self):
        """Start the transcription process."""
        if self.is_running:
            logger.warning("Xenova transcriber is already running")
            return
        
        self.is_running = True
        self.transcription_thread = threading.Thread(
            target=self._transcribe_audio_stream,
            daemon=True
        )
        self.transcription_thread.start()
        logger.info("Xenova transcriber started")
    
    def stop(self):
        """Stop the transcription process."""
        if not self.is_running:
            logger.warning("Xenova transcriber is not running")
            return
        
        self.is_running = False
        if self.transcription_thread:
            self.transcription_thread.join(timeout=2.0)
        
        # Clear any remaining results
        while not self.results_queue.empty():
            try:
                self.results_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Xenova transcriber stopped")
    
    def get_latest_result(self, block=False, timeout=None):
        """
        Get the latest transcription result.
        
        Args:
            block: If True, block until a new result is available
            timeout: Maximum time to wait if blocking
            
        Returns:
            The latest transcription result as a dict
        """
        try:
            if block:
                result = self.results_queue.get(block=True, timeout=timeout)
                self.latest_result = result
            else:
                # Non-blocking: check if there's a new result
                if not self.results_queue.empty():
                    result = self.results_queue.get_nowait()
                    self.latest_result = result
                
            return self.latest_result
                
        except queue.Empty:
            return self.latest_result
    
    def _transcribe_audio_stream(self):
        """
        Capture and transcribe audio in real-time using the system microphone or audio device.
        This runs in a separate thread.
        """
        try:
            import sounddevice as sd
            from transformers import pipeline
            
            # Initialize the whisper pipeline
            logger.info(f"Loading Xenova Whisper model: {self.model_name}")
            transcriber = pipeline(
                "automatic-speech-recognition", 
                model=f"openai/whisper-{self.model_name}",
                device=self.device
            )
            
            # Set up audio parameters
            sample_rate = 16000  # Whisper expects 16kHz audio
            chunk_duration_sec = self.chunk_duration_ms / 1000
            frames_per_chunk = int(sample_rate * chunk_duration_sec)
            
            # Buffer for audio data
            buffer = []
            all_text = ""
            segments = []
            
            def audio_callback(indata, frames, time, status):
                """Callback for the audio stream to collect audio chunks."""
                if status:
                    logger.warning(f"Audio stream status: {status}")
                
                # Convert to mono if needed and store the audio data
                if indata.shape[1] > 1:
                    mono_data = indata.mean(axis=1)
                else:
                    mono_data = indata.flatten()
                
                buffer.append(mono_data)
                
                # If we have enough data, process it
                if len(buffer) * frames_per_chunk > sample_rate * 2:  # At least 2 seconds
                    audio_data = np.concatenate(buffer)
                    
                    # Process in a separate thread to avoid blocking the audio stream
                    threading.Thread(
                        target=self._process_audio_chunk,
                        args=(audio_data, transcriber, sample_rate, all_text, segments),
                        daemon=True
                    ).start()
                    
                    # Clear the buffer but keep a small overlap
                    overlap_samples = int(sample_rate * 0.5)  # 0.5 seconds overlap
                    if len(buffer) > 0 and len(buffer[-1]) > overlap_samples:
                        buffer = [buffer[-1][-overlap_samples:]]
                    else:
                        buffer = []
            
            # Start the audio stream
            logger.info("Starting audio stream for Xenova transcription")
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                callback=audio_callback,
                blocksize=frames_per_chunk
            ):
                # Keep running until stopped
                while self.is_running:
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Error in Xenova transcription thread: {e}")
            self.is_running = False
    
    def _process_audio_chunk(self, audio_data, transcriber, sample_rate, all_text, segments):
        """Process an audio chunk and add the result to the queue."""
        try:
            # Perform transcription
            result = transcriber(
                {"array": audio_data, "sampling_rate": sample_rate},
                generate_kwargs={"language": self.language} if self.language else {}
            )
            
            # Extract text
            text = result.get("text", "").strip()
            
            if text:
                # Update the accumulated text and segments
                all_text += text + " "
                segments.append({
                    "text": text,
                    "start": time.time() - len(audio_data) / sample_rate,
                    "end": time.time()
                })
                
                # Put the result in the queue
                self.results_queue.put({
                    "text": all_text.strip(),
                    "segments": segments,
                    "timestamp": time.time()
                })
                
                logger.debug(f"Transcribed chunk: {text}")
        
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    def transcribe_file(self, file_path):
        """
        Transcribe an audio or video file.
        
        Args:
            file_path: Path to the audio or video file
            
        Returns:
            The transcribed text
        """
        try:
            from transformers import pipeline
            
            # Check if the file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Load the ASR pipeline
            logger.info(f"Loading Xenova Whisper model for file transcription: {self.model_name}")
            transcriber = pipeline(
                "automatic-speech-recognition", 
                model=f"openai/whisper-{self.model_name}",
                device=self.device
            )
            
            # Extract audio if the file is a video
            if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
                logger.info(f"Extracting audio from video file: {file_path}")
                audio_path = self._extract_audio_from_video(file_path)
            else:
                audio_path = file_path
            
            # Perform transcription
            logger.info(f"Transcribing file: {audio_path}")
            result = transcriber(
                audio_path,
                generate_kwargs={"language": self.language} if self.language else {}
            )
            
            # Clean up temporary audio file if created
            if audio_path != file_path:
                os.remove(audio_path)
            
            # Extract and return the transcribed text
            text = result.get("text", "").strip()
            logger.info(f"Transcription completed. Length: {len(text)}")
            return text
            
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            raise
    
    def _extract_audio_from_video(self, video_path):
        """
        Extract audio from a video file using ffmpeg.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the extracted audio file
        """
        # Create a temporary file for the audio
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        # Use ffmpeg to extract audio
        try:
            logger.info(f"Extracting audio from video: {video_path} to {audio_path}")
            subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-vn", "-acodec", "pcm_s16le", 
                    "-ar", "16000", "-ac", "1",
                    audio_path
                ],
                check=True,
                capture_output=True
            )
            return audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            os.remove(audio_path)
            raise Exception(f"Failed to extract audio: {e}")

# Function to use in routes
def transcribe_with_xenova(file_path, language=None, model_name="base"):
    """
    Transcribe a file using the Xenova engine.
    
    Args:
        file_path: Path to the audio or video file
        language: Language code (e.g., "en") or None for auto-detection
        model_name: Whisper model to use
        
    Returns:
        The transcribed text
    """
    transcriber = XenovaTranscriber(model_name=model_name, language=language)
    return transcriber.transcribe_file(file_path)
