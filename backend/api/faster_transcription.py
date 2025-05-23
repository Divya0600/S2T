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
from collections import deque
import difflib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class FasterTranscriber:
    """
    A class to handle real-time audio transcription using the faster-whisper model.
    Improved version with better deduplication and timestamp management.
    """
    def __init__(
        self,
        model_name: str = "base",
        language: Optional[str] = None,
        chunk_duration_ms: int = 3000,  # Reduced chunk size for better real-time performance
        sample_rate: int = 16000,
        channels: int = 1,
        format_type: int = pyaudio.paInt16,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        compute_type: str = "int8",
        device_type: str = "cpu"
    ):
        """
        Initialize the real-time transcriber using faster-whisper.
        
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
        
        # Audio buffer management
        self.audio_buffer = deque(maxlen=10)  # Keep last 10 chunks for context
        self.processed_audio_duration = 0.0  # Track total processed time
        self.last_transcript = ""  # Keep track of last complete transcript
        self.all_segments = []  # Store all confirmed segments
        self.pending_text = ""  # Store text that might be incomplete
        
        # Deduplication settings
        self.similarity_threshold = 0.8  # Threshold for text similarity
        self.min_segment_length = 3  # Minimum words for a valid segment
        
        # Audio parameters
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self.chunk_bytes = self.chunk_samples * channels * 2
        self.chunk_duration_sec = chunk_duration_ms / 1000.0
        
        # State flags
        self.running = False
        self.paused = False
        self.device_index = None
        
        # Threading and queue for audio processing
        self.audio_queue = queue.Queue(maxsize=5)  # Limit queue size
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
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        # Use difflib to calculate similarity
        similarity = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        return similarity
    
    def _is_duplicate_segment(self, new_text: str, new_start: float, new_end: float) -> bool:
        """Check if a segment is a duplicate of existing segments."""
        if len(new_text.split()) < self.min_segment_length:
            return True  # Skip very short segments
        
        for segment in self.all_segments:
            # Check text similarity
            similarity = self._text_similarity(new_text, segment['text'])
            
            # Check temporal overlap
            time_overlap = (
                max(0, min(new_end, segment['end']) - max(new_start, segment['start']))
                / min(new_end - new_start, segment['end'] - segment['start'])
            )
            
            # Consider duplicate if high similarity and some time overlap
            if similarity > self.similarity_threshold and time_overlap > 0.3:
                logger.debug(f"Duplicate detected: similarity={similarity:.2f}, overlap={time_overlap:.2f}")
                return True
        
        return False
    
    def _merge_overlapping_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge segments that have significant overlap."""
        if not segments:
            return []
        
        merged = []
        for segment in sorted(segments, key=lambda x: x['start']):
            if not merged:
                merged.append(segment)
                continue
            
            last_segment = merged[-1]
            
            # Check if segments overlap significantly
            overlap_start = max(segment['start'], last_segment['start'])
            overlap_end = min(segment['end'], last_segment['end'])
            overlap_duration = max(0, overlap_end - overlap_start)
            
            segment_duration = segment['end'] - segment['start']
            last_duration = last_segment['end'] - last_segment['start']
            
            # If overlap is significant, merge the segments
            overlap_ratio = overlap_duration / min(segment_duration, last_duration)
            
            if overlap_ratio > 0.5:  # 50% overlap threshold
                # Merge by extending the time range and choosing the longer text
                merged_text = segment['text'] if len(segment['text']) > len(last_segment['text']) else last_segment['text']
                
                merged[-1] = {
                    'start': min(segment['start'], last_segment['start']),
                    'end': max(segment['end'], last_segment['end']),
                    'text': merged_text,
                    'confidence': max(segment.get('confidence', 0), last_segment.get('confidence', 0))
                }
                logger.debug(f"Merged overlapping segments: {overlap_ratio:.2f} overlap")
            else:
                merged.append(segment)
        
        return merged
    
    def start(self):
        """Start audio capture and transcription using the default input device."""
        if self.running:
            logger.warning("Transcription is already running")
            return
        
        logger.info("Starting real-time transcription")
        self.running = True
        self.paused = False
        
        # Reset state
        self.audio_buffer.clear()
        self.processed_audio_duration = 0.0
        self.last_transcript = ""
        self.all_segments = []
        self.pending_text = ""
        
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
            
        self.processing_thread = threading.Thread(target=self._process_audio_thread, daemon=True)
        self.processing_thread.start()
        
        logger.info("Audio stream and processing thread started")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for the PyAudio stream."""
        if not self.paused and self.running:
            try:
                # Convert audio data to numpy array
                audio_np = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Add to queue, skip if queue is full to prevent backing up
                try:
                    self.audio_queue.put_nowait(audio_np)
                except queue.Full:
                    logger.warning("Audio queue full, dropping frame")
                    
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
        
        return (in_data, pyaudio.paContinue)
    
    def _process_audio_thread(self):
        """Thread for processing audio chunks and running transcription."""
        logger.info("Audio processing thread started")
        
        last_process_time = time.time()
        
        while self.running:
            try:
                # Get audio chunk with timeout
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Add to buffer
                self.audio_buffer.append(audio_chunk)
                
                # Process every few chunks or when buffer is getting full
                current_time = time.time()
                if (current_time - last_process_time > 2.0) or len(self.audio_buffer) >= 3:
                    self._process_accumulated_audio()
                    last_process_time = current_time
                
                self.audio_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in audio processing thread: {e}")
                time.sleep(0.1)
        
        logger.info("Audio processing thread stopped")
    
    def _process_accumulated_audio(self):
        """Process accumulated audio chunks."""
        if not self.audio_buffer:
            return
        
        try:
            # Combine all audio chunks in buffer
            combined_audio = np.concatenate(list(self.audio_buffer))
            
            # Skip if audio is too quiet
            if np.abs(combined_audio).max() < 0.01:
                return
            
            # Transcribe the combined audio
            segments, info = self.model.transcribe(
                combined_audio,
                language=self.language,
                beam_size=3,  # Reduced for faster processing
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=100
                )
            )
            
            # Process segments
            new_segments = []
            buffer_duration = len(combined_audio) / self.sample_rate
            
            for segment in segments:
                # Adjust timestamps to global time
                adjusted_start = self.processed_audio_duration + segment.start
                adjusted_end = self.processed_audio_duration + segment.end
                
                segment_text = segment.text.strip()
                if not segment_text or len(segment_text.split()) < 2:
                    continue
                
                # Check if this is a duplicate
                if not self._is_duplicate_segment(segment_text, adjusted_start, adjusted_end):
                    new_segments.append({
                        'start': adjusted_start,
                        'end': adjusted_end,
                        'text': segment_text,
                        'confidence': getattr(segment, 'avg_logprob', 0.0)
                    })
            
            # Merge with existing segments and remove overlaps
            if new_segments:
                # Add new segments to all segments
                self.all_segments.extend(new_segments)
                
                # Merge overlapping segments
                self.all_segments = self._merge_overlapping_segments(self.all_segments)
                
                # Sort by start time
                self.all_segments.sort(key=lambda x: x['start'])
                
                # Create result
                full_text = " ".join(seg['text'] for seg in self.all_segments)
                
                result = {
                    "text": full_text,
                    "segments": new_segments,  # Only send new segments
                    "all_segments": self.all_segments,  # Send all for context
                    "timestamp": time.time()
                }
                
                # Put result in queue and call callback
                self.result_queue.put(result)
                
                if self.callback:
                    self.callback(full_text, result)
                
                logger.info(f"Processed {len(new_segments)} new segments, total: {len(self.all_segments)}")
            
            # Update processed duration (move forward by half the buffer to maintain overlap)
            advance_duration = buffer_duration * 0.6  # 60% advance for some overlap
            self.processed_audio_duration += advance_duration
            
            # Keep only the last chunk for next iteration (maintain some overlap)
            if len(self.audio_buffer) > 1:
                self.audio_buffer = deque([self.audio_buffer[-1]], maxlen=10)
            
        except Exception as e:
            logger.error(f"Error processing accumulated audio: {e}")
    
    def stop(self):
        """Stop the audio stream and cleanup resources."""
        if not self.running:
            logger.warning("Transcription is not running")
            return
            
        logger.info("Stopping transcription")
        
        # Set flag to stop threads
        self.running = False
        self.paused = False
        
        # Process any remaining audio
        if self.audio_buffer:
            try:
                self._process_accumulated_audio()
            except Exception as e:
                logger.error(f"Error processing final audio: {e}")
        
        # Close and clean up audio resources
        try:
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop_stream()
                self.stream.close()
                logger.info("Audio stream stopped")
            
            if hasattr(self, 'audio') and self.audio:
                self.audio.terminate()
                logger.info("PyAudio terminated")
                    
        except Exception as e:
            logger.error(f"Error cleaning up audio resources: {e}")
            
        # Clear queues
        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
        except:
            pass
            
        # Wait for processing thread to complete
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=3.0)
            logger.info("Processing thread joined")
        
        logger.info("Transcription stopped successfully")
    
    def get_latest_result(self, block: bool = False, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
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
                return self.result_queue.get(block=True, timeout=timeout)
            else:
                return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def pause(self):
        """Pause transcription without stopping the audio stream."""
        logger.info("Pausing transcription")
        self.paused = True
    
    def resume(self):
        """Resume transcription after pausing."""
        logger.info("Resuming transcription")
        self.paused = False
    
    def reset(self):
        """Reset the transcriber state, clearing any buffered audio and results."""
        with self.lock:
            logger.info("Resetting transcriber state")
            
            # Clear buffers
            self.audio_buffer.clear()
            self.all_segments = []
            self.last_transcript = ""
            self.pending_text = ""
            self.processed_audio_duration = 0.0
            
            # Clear queues
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                except queue.Empty:
                    break
                    
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    break
    
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


# Singleton instance for application-wide use
_transcriber_instance = None

def get_transcriber(config=None):
    """
    Get or create the global transcriber instance.
    
    Args:
        config: Optional configuration dictionary to initialize the transcriber
        
    Returns:
        The global FasterTranscriber instance
    """
    global _transcriber_instance
    
    if _transcriber_instance is None and config is not None:
        # Handle device parameter for backward compatibility
        if 'device_index' in config and 'device' not in config:
            config['device'] = config.pop('device_index')
        
        # Remove 'engine' parameter as it's not supported by FasterTranscriber
        config_copy = config.copy()
        config_copy.pop('engine', None)
            
        _transcriber_instance = FasterTranscriber(**config_copy)
    
    return _transcriber_instance