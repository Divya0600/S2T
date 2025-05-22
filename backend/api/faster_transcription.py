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
        
        self.device_index = None # Will be set in start() to default input device
        
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
            # Add audio data to the queue
            try:
                # Process audio data in smaller chunks for more frequent updates
                audio_np = np.frombuffer(in_data, dtype=np.int16)
                
                # Add audio data to the queue
                self.audio_queue.put(audio_np)
                
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
        return (in_data, pyaudio.paContinue)
        
    def _process_audio_thread(self):
        """Thread for processing audio chunks and running transcription."""
        logger.info("Audio processing thread started")
        
        # Following the reference implementation pattern from whisper-live-transcription-main
        # Buffer for accumulating audio in a sliding window approach
        audio_buffer = []
        max_buffer_chunks = 6  # Maximum seconds of audio to keep buffered
        
        # Stats for monitoring performance
        stats = {"transcription_times": [], "overall_times": []}
        
        # Keep track of all accumulated segments for continuous transcription
        all_segments = []
        transcript_history = ""
        
        # Time tracking
        audio_time_offset = 0  # Keep track of time offsets for accurate timestamps
        
        while self.running:
            if not self.paused:
                try:
                    # Get the next audio chunk from the queue with timeout
                    audio_data = self.audio_queue.get(block=True, timeout=0.5)
                    
                    # Start timing overall processing
                    overall_start = time.time()
                    
                    # Add to buffer and maintain max size using sliding window approach
                    audio_buffer.append(audio_data)
                    if len(audio_buffer) > max_buffer_chunks:
                        # If buffer is full, remove oldest chunk
                        audio_buffer.pop(0)
                        # Update time offset for accurate timestamps
                        audio_time_offset += self.chunk_duration_ms / 1000
                    
                    # Combine all audio chunks in buffer for better context
                    # This is critical for good transcription quality
                    if len(audio_buffer) > 0:
                        # Convert int16 to float32 and normalize
                        combined_audio = np.concatenate(audio_buffer)
                        if combined_audio.dtype == np.int16:
                            combined_audio = combined_audio.astype(np.float32) / 32768.0
                        
                        # Start timing just the transcription part
                        transcription_start = time.time()
                        
                        # Transcribe with optimized settings for CPU
                        segments, info = self.model.transcribe(
                            combined_audio,
                            beam_size=3,  # Lower beam size for faster processing
                            language=self.language,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500)
                        )
                        
                        # Record transcription time
                        transcription_time = time.time() - transcription_start
                        stats["transcription_times"].append(transcription_time)
                        
                        # Process segments and build result
                        segments_list = []
                        text = ""
                        
                        for segment in segments:
                            # Apply correct time offset to timestamps
                            start = segment.start + audio_time_offset
                            end = segment.end + audio_time_offset
                            clean_text = re.sub(r'\[.*?\]|\(.*?\)', '', segment.text).strip()
                            
                            if clean_text:  # Only add non-empty segments
                                segments_list.append({
                                    "start": start,
                                    "end": end,
                                    "text": clean_text
                                })
                                text += clean_text + " "
                        
                        # Calculate overall processing time
                        overall_time = time.time() - overall_start
                        stats["overall_times"].append(overall_time)
                        
                        # Log statistics periodically
                        if len(stats["overall_times"]) % 10 == 0:
                            avg_trans = sum(stats["transcription_times"]) / len(stats["transcription_times"])
                            avg_overall = sum(stats["overall_times"]) / len(stats["overall_times"])
                            logger.info(f"Avg transcription: {avg_trans:.3f}s, Overall: {avg_overall:.3f}s")
                        
                        # Only create a result when we have detected speech
                        if text.strip():
                            # Update the all_segments list with new segments
                            all_segments.extend(segments_list)
                            
                            # Create the result with ALL accumulated segments for continuous transcription
                            result = {
                                "text": text.strip(),
                                "segments": segments_list,  # Send just the new segments
                                "all_segments": all_segments,  # But include all history for frontend
                                "language": info.language if hasattr(info, 'language') else None,
                                "timestamp": time.time()
                            }
                            
                            # Add to result queue and call callback if provided
                            self.result_queue.put(result)
                            if self.callback:
                                self.callback(text, result)
                    
                    # Mark task as done
                    self.audio_queue.task_done()
                    
                except queue.Empty:
                    # No audio available, just continue
                    pass
                except Exception as e:
                    logger.error(f"Error in audio processing: {str(e)}")
            else:
                # Paused state - just wait briefly
                time.sleep(0.1)
        
        logger.info("Audio processing thread stopped")
                
    
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
        
    def _transcribe_internal(self, file_path):
        try:
            logger.info("Starting transcription with faster-whisper...")
            segments, info = self.model.transcribe(
                file_path,
                language=self.language or "en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            # Convert generator to list to avoid generator issues across threads
            segments_list = list(segments)
            logger.info(f"Transcription completed. Segments: {len(segments_list)}")
            return segments_list, info
        except Exception as e:
            logger.error(f"Error in _transcribe_internal: {str(e)}", exc_info=True)
            raise
            
    async def transcribe_file(self, file_path: str):
        """
        Transcribe an audio or video file using the loaded model.
        
        Args:
            file_path: Path to the audio or video file to transcribe
            
        Returns:
            A dictionary containing both the full transcript and timestamped segments
            
        Raises:
            Exception: If there's an error during transcription
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError("Cannot transcribe an empty file")
        
        try:
            # Use the loaded model to transcribe the file
            # Run the synchronous transcribe method in a thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            
            # Run the blocking transcribe call in a thread pool
            logger.info("Running transcription in thread pool...")
            segments, info = await loop.run_in_executor(None, lambda: self._transcribe_internal(file_path))
            
            if not segments:
                logger.warning("No transcription segments were returned")
                return {
                    "text": "",
                    "segments": []
                }
            
            # Format segments for frontend
            formatted_segments = []
            for segment in segments:
                formatted_segments.append({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end
                })
            
            # Assemble the complete transcript
            transcript = " ".join(segment.text for segment in segments)
            logger.info(f"Transcription completed. Text length: {len(transcript)} characters")
            
            return {
                "text": transcript.strip(),
                "segments": formatted_segments
            }
            
        except Exception as e:
            logger.error(f"Error transcribing file {file_path}: {str(e)}", exc_info=True)
            raise Exception(f"Failed to transcribe file: {str(e)}") from e

# Singleton instance for application-wide use
_transcriber_instance = None

def get_transcriber(config=None):
    """
    Get or create the global transcriber instance.
    
    Args:
        config: Optional configuration dictionary to initialize the transcriber
        
    Returns:
        The global FasterTranscriber instance
        
    Raises:
        Exception: If there's an error initializing the transcriber
    """
    global _transcriber_instance
    
    if _transcriber_instance is not None:
        return _transcriber_instance
        
    try:
        if config is None:
            config = {}
            
        # Set default values if not provided in config
        model_name = config.get('model_name', 'base')
        language = config.get('language', 'en')
        device_type = config.get('device_type', 'cpu')
        compute_type = config.get('compute_type', 'int8')
        
        logger.info(f"Initializing FasterTranscriber with model: {model_name}, device: {device_type}")
        
        _transcriber_instance = FasterTranscriber(
            model_name=model_name,
            language=language,
            device_type=device_type,
            compute_type=compute_type
        )
        
        # Verify the model was initialized successfully
        logger.info("Verifying transcriber initialization...")
        if _transcriber_instance.model is not None:
            logger.info(f"Transcriber initialized successfully with model: {model_name}")
        else:
            raise Exception("Model initialization failed, model is None")
        
        return _transcriber_instance
        
    except Exception as e:
        logger.error(f"Failed to initialize FasterTranscriber: {str(e)}", exc_info=True)
        _transcriber_instance = None  # Reset instance to allow retry
        raise Exception(f"Failed to initialize transcriber: {str(e)}") from e
