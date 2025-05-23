import os
import queue
import threading
import time
import logging
import numpy as np
import pyaudio
import signal
import asyncio
import re
import tempfile
import wave
from typing import Optional, Dict, Any, List, Callable, Union
from faster_whisper import WhisperModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class FasterTranscriber:
    """
    A class to handle real-time audio transcription using the faster-whisper model.
    """
    def __init__(
        self,
        model_name: str = "base",
        language: Optional[str] = None,
        chunk_duration_ms: int = 5000,
        sample_rate: int = 16000,
        channels: int = 1,
        format_type: int = pyaudio.paInt16,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        compute_type: str = "int8",
        device_type: str = "cpu"
    ):
        """
        Initialize the real-time transcriber using faster-whisper.
        Simplified to use default audio input device.
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3')
            language: Language code (e.g., 'en', 'fr') or None for auto-detection
            chunk_duration_ms: Duration of each audio chunk in milliseconds
            sample_rate: Audio sample rate (16000 Hz is recommended)
            channels: Number of audio channels (1 for mono)
            format_type: PyAudio format type
            callback: Optional callback function for receiving transcription results
            compute_type: Compute type for faster-whisper ('float16', 'int8', etc.)
            device_type: Device type ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.language = language
        self.chunk_duration_ms = chunk_duration_ms
        self.sample_rate = sample_rate
        self.channels = channels
        self.format_type = format_type
        self.callback = callback
        self.compute_type = compute_type
        self.device_type = device_type
        
        # Track audio processing state
        self.audio_offset = 0.0  # Track total processed audio time
        self.last_processed_time = 0.0
        self.last_audio_chunk = None
        self.device_index = None  # Will be set in start() to default input device
        
        # Audio parameters
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self.chunk_bytes = self.chunk_samples * channels * 2  # 2 bytes per sample for paInt16
        self.chunk_duration_sec = chunk_duration_ms / 1000.0
        
        # State flags
        self.running = False
        self.paused = False
        
        # Threading and queue for audio processing
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.lock = threading.Lock()
        
        # Load the faster-whisper model
        logger.info(f"Loading faster-whisper model: {model_name} with compute_type {compute_type} on {device_type}")
        try:
            cpu_threads = 4
            if device_type == "cpu":
                logger.info(f"Optimizing for CPU with {cpu_threads} threads")
                self.model = WhisperModel(
                    model_name, 
                    device=device_type, 
                    compute_type=compute_type,
                    cpu_threads=cpu_threads
                )
            else:
                self.model = WhisperModel(
                    model_name, 
                    device=device_type, 
                    compute_type=compute_type
                )
            logger.info(f"faster-whisper model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load faster-whisper model: {e}")
            raise

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals for graceful shutdown."""
        logger.info(f"Received signal {sig}, stopping transcription...")
        self.stop()
    
    def start(self):
        """Start audio capture and transcription using the default input device."""
        if self.running:
            logger.warning("Transcription is already running")
            return
            
        self.running = True
        self.paused = False
        
        self.audio = pyaudio.PyAudio()
        
        try:
            default_device_info = self.audio.get_default_input_device_info()
            self.device_index = default_device_info.get('index')
            logger.info(f"Using default input device: {default_device_info.get('name')} (Index: {self.device_index})")
        except Exception as e:
            logger.error(f"Could not get default input device: {e}")
            self.audio.terminate()
            self.running = False
            raise ValueError("Could not get default input device.")

        if self.device_index is None:
            logger.error("Default input device index is None. Cannot start audio stream.")
            self.audio.terminate()
            self.running = False
            raise ValueError("Default input device index is None.")

        try:
            self.stream = self.audio.open(
                format=self.format_type,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_samples,
                input_device_index=self.device_index,
                stream_callback=self._audio_callback
            )
            logger.info(f"Started audio stream with device index {self.device_index}")
        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}")
            self.audio.terminate()
            self.running = False
            raise ValueError(f"Failed to open audio stream with device index {self.device_index}.")
            
        self.processing_thread = threading.Thread(target=self._process_audio_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        stream_info = self.stream.get_input_latency()
        logger.info(f"Audio stream started. Input latency: {stream_info} seconds")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for the PyAudio stream."""
        if not self.paused:
            try:
                # Convert audio data to numpy array and add to queue
                audio_np = np.frombuffer(in_data, dtype=np.int16)
                self.audio_queue.put(audio_np)
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
        return (in_data, pyaudio.paContinue)
    
    def _process_audio_chunk(self, audio_data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process a single audio chunk and return transcribed segments with adjusted timestamps.
        
        Args:
            audio_data: Numpy array containing the audio data (int16 format)
            
        Returns:
            List of transcribed segments with adjusted timestamps
        """
        try:
            # Calculate chunk duration
            chunk_duration = len(audio_data) / self.sample_rate
            
            # Transcribe the chunk
            segments, _ = self.model.transcribe(
                audio_data,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Process segments and adjust timestamps
            result_segments = []
            for segment in segments:
                result_segments.append({
                    'start': self.audio_offset + segment.start,
                    'end': self.audio_offset + segment.end,
                    'text': segment.text.strip(),
                    'confidence': segment.avg_logprob
                })
            
            # Update the audio offset for the next chunk
            self.audio_offset += chunk_duration
            
            return result_segments
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return []
    
    def _process_audio_thread(self):
        """Thread for processing audio chunks and running transcription."""
        logger.info("Audio processing thread started")
        
        # Buffer for accumulating audio in a sliding window approach
        audio_buffer = []
        max_buffer_chunks = 6  # Maximum seconds of audio to keep buffered (~30s with 5s chunks)
        
        # Stats for monitoring performance
        stats = {"transcription_times": [], "overall_times": []}
        
        # Keep track of all transcribed segments
        all_segments = []
        
        # Time tracking
        last_processed_time = time.time()
        
        while self.running:
            if not self.paused:
                try:
                    # Get the next audio chunk from the queue with timeout
                    audio_data = self.audio_queue.get(block=True, timeout=0.5)
                    
                    # Add new audio to buffer
                    audio_buffer.append(audio_data)
                    
                    # Process the buffer if we have enough data or if we're at the end of the stream
                    if len(audio_buffer) >= max_buffer_chunks or self.audio_queue.empty():
                        # Concatenate audio chunks for processing
                        combined_audio = np.concatenate(audio_buffer) if len(audio_buffer) > 1 else audio_buffer[0]
                        
                        # Process the audio chunk
                        start_time = time.time()
                        segments = self._process_audio_chunk(combined_audio)
                        processing_time = time.time() - start_time
                        
                        if segments:
                            # Add new segments to our collection
                            all_segments.extend(segments)
                            
                            # Update the transcript text
                            transcript_text = " ".join(seg["text"] for seg in all_segments if seg["text"])
                            
                            # Prepare the result
                            result = {
                                "text": transcript_text,
                                "segments": all_segments,
                                "is_partial": False,
                                "processing_time": processing_time
                            }
                            
                            # Put the result in the result queue
                            self.result_queue.put(result)
                            
                            # Call the callback if provided
                            if self.callback:
                                self.callback(transcript_text, result)
                            
                            # Update stats
                            stats["transcription_times"].append(processing_time)
                            stats["overall_times"].append(time.time() - last_processed_time)
                            last_processed_time = time.time()
                            
                            # Log some stats occasionally
                            if len(stats["transcription_times"]) % 10 == 0:
                                avg_transcribe = sum(stats["transcription_times"][-10:]) / 10
                                avg_overall = sum(stats["overall_times"][-10:]) / 10
                                logger.info(
                                    f"Avg transcription: {avg_transcribe:.2f}s, "
                                    f"Avg overall: {avg_overall:.2f}s, "
                                    f"Buffer size: {len(audio_buffer)}"
                                )
                        
                        # Clear the buffer, keeping the last chunk for overlap if needed
                        audio_buffer = audio_buffer[-1:] if audio_buffer else []
                    
                    self.audio_queue.task_done()
                    
                except queue.Empty:
                    # No audio data available, just continue
                    continue
                    
                except Exception as e:
                    logger.error(f"Error in audio processing thread: {e}")
                    # Add a small delay to prevent tight loops on error
                    time.sleep(0.1)
            else:
                # Paused state - just wait briefly
                time.sleep(0.1)
    
    def stop(self):
        """Stop the audio stream and cleanup resources."""
        if not self.running:
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
    
    def get_latest_result(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Get the latest transcription result.
        
        Args:
            block: If True, block until a result is available
            timeout: Maximum time to wait for a result if block is True
                
        Returns:
            The latest transcription result or None if no result is available
        """
        try:
            if block:
                result = self.result_queue.get(block=True, timeout=timeout)
                if result and self.callback:
                    self.callback(result["text"], result)
                return result
            else:
                result = self.result_queue.get_nowait()
                if result and self.callback:
                    self.callback(result["text"], result)
                return result
        except queue.Empty:
            return None
    
    def reset(self):
        """
        Reset the transcriber state, clearing any buffered audio and results.
        This is useful when starting a new transcription session.
        """
        with self.lock:
            # Clear any pending audio data
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                except queue.Empty:
                    break
                    
            # Clear any pending results
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    break
                
            # Reset state variables
            self.audio_offset = 0.0
            self.last_processed_time = 0.0
            self.last_audio_chunk = None
    
    def _transcribe_internal(self, file_path):
        """Internal method to transcribe a file."""
        try:
            logger.info(f"Starting transcription of {file_path}")
            segments, info = self.model.transcribe(
                file_path,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Convert generator to list and process segments
            segments_list = []
            for segment in segments:
                segments_list.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'confidence': segment.avg_logprob
                })
            
            return segments_list, info
            
        except Exception as e:
            logger.error(f"Error in _transcribe_internal: {e}", exc_info=True)
            raise
    
    async def transcribe_file(self, file_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio or video file using the loaded model.
        
        Args:
            file_path: Path to the audio or video file to transcribe
            
        Returns:
            Dictionary containing the transcription results
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Transcribe the file
            segments, info = await asyncio.get_event_loop().run_in_executor(
                None,
                self._transcribe_internal,
                file_path
            )
            
            # Combine all text
            text = " ".join(seg["text"] for seg in segments if seg["text"])
            
            return {
                'text': text,
                'segments': segments,
                'language': info.language if hasattr(info, 'language') else None,
                'language_probability': getattr(info, 'language_probability', None)
            }
            
        except Exception as e:
            logger.error(f"Error in transcribe_file: {e}", exc_info=True)
            raise
