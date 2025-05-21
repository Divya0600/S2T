import os
import tempfile
import whisper
import torch
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class WhisperTranscriber:
    def __init__(self, model_name: str = "base"):
        """
        Initialize the WhisperTranscriber with the specified model.
        
        Args:
            model_name: The name of the Whisper model to use (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the Whisper model."""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            try:
                # Load model with the download directory set to MODELS_DIR
                self.model = whisper.load_model(
                    self.model_name, 
                    device=self.device,
                    download_root=MODELS_DIR
                )
                logger.info(f"Successfully loaded Whisper model: {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {str(e)}")
                raise
    
    def transcribe(self, audio_file: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe the given audio file with timestamps.
        
        Args:
            audio_file: Path to the audio file
            language: Optional language code
            
        Returns:
            Dictionary containing the transcript and segments with timestamps
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Set transcription options
            options = {
                "task": "transcribe",
                "verbose": True,
            }
            
            if language:
                options["language"] = language
            
            # Transcribe with timestamps
            result = self.model.transcribe(audio_file, **options)
            
            # Process segments to extract text and timestamps
            segments_with_timestamps = []
            for segment in result.get("segments", []):
                segments_with_timestamps.append({
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"]
                })
            
            return {
                "text": result.get("text", ""),
                "segments": segments_with_timestamps
            }
            
        except Exception as e:
            logger.error(f"Error transcribing with Whisper: {str(e)}")
            raise

# Singleton instance for reuse
_transcriber_instance = None

def get_whisper_transcriber(model_name: str = "base") -> WhisperTranscriber:
    """
    Get or create a WhisperTranscriber instance.
    
    Args:
        model_name: The name of the Whisper model to use
        
    Returns:
        WhisperTranscriber instance
    """
    global _transcriber_instance
    
    if _transcriber_instance is None or _transcriber_instance.model_name != model_name:
        _transcriber_instance = WhisperTranscriber(model_name)
        
    return _transcriber_instance

def transcribe_audio_chunk(audio_data: bytes, model_name: str = "base", language: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe an audio chunk with Whisper.
    
    Args:
        audio_data: Audio data as bytes
        model_name: The Whisper model to use
        language: Optional language code
        
    Returns:
        Dictionary with transcript and segments with timestamps
    """
    # Get transcriber instance
    transcriber = get_whisper_transcriber(model_name)
    
    # Create a temporary file for the audio data
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_data)
        temp_file_path = temp_file.name
    
    try:
        # Transcribe the audio file
        result = transcriber.transcribe(temp_file_path, language)
        return result
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# --- Whisper.cpp Integration for Live Transcription ---
import subprocess
import re

# Paths retrieved from memory/setup script
# Ensure these paths are correct for your environment.
WHISPER_CPP_BASE_DIR = r"C:\Users\divya.eesarla\Desktop\Speech Local - Copy\whisper.cpp"
# Path to the main whisper.cpp executable (CMake build path)
WHISPER_CPP_MAIN_EXE = os.path.join(WHISPER_CPP_BASE_DIR, "build", "bin", "Release", "main.exe")
# Path to the whisper.cpp model (ggml-base.en.bin)
WHISPER_CPP_MODEL_PATH = os.path.join(WHISPER_CPP_BASE_DIR, "models", "ggml-base.en.bin")

logger.info(f"Whisper.cpp main executable path: {WHISPER_CPP_MAIN_EXE}")
logger.info(f"Whisper.cpp model path: {WHISPER_CPP_MODEL_PATH}")

if not os.path.isfile(WHISPER_CPP_MAIN_EXE):
    logger.error(f"FATAL: Whisper.cpp main.exe not found at {WHISPER_CPP_MAIN_EXE}. Please run setup_whisper_cpp.bat.")
if not os.path.isfile(WHISPER_CPP_MODEL_PATH):
    logger.error(f"FATAL: Whisper.cpp model not found at {WHISPER_CPP_MODEL_PATH}. Please ensure it was downloaded.")

def parse_whisper_cpp_output(output: str) -> List[Dict[str, Any]]:
    """
    Parse the stdout of whisper.cpp (main.exe) to extract timestamped segments.
    Expected format: [HH:MM:SS.mmm --> HH:MM:SS.mmm] Text segment
    """
    segments = []
    # Regex to capture timestamps and text. Handles potential extra spaces.
    pattern = re.compile(r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)")
    for line in output.strip().split('\n'):
        match = pattern.match(line)
        if match:
            start_time, end_time, text = match.groups()
            segments.append({
                "start": start_time,
                "end": end_time,
                "text": text.strip()
            })
            logger.debug(f"Parsed segment: {{'start': '{start_time}', 'end': '{end_time}', 'text': '{text.strip()}'}}")
        elif line.strip() and not line.startswith("whisper_init_from_file") and not line.startswith("log_mel_filter_bank") and not line.startswith("encode_segment") and not line.startswith("decode_segment") and not line.startswith("prompt") and not line.startswith("system_info"):
            # Log non-empty lines that don't match the expected segment format and are not known info lines
            logger.warning(f"Non-matching or info line from whisper.cpp stdout: {line}")
    return segments

def transcribe_audio_chunk_cpp(audio_data: bytes, language: str = "en") -> Dict[str, Any]:
    """
    Transcribe an audio chunk using whisper.cpp's main.exe.

    Args:
        audio_data: Audio data as bytes (expected to be WAV format compatible).
        language: Language code for transcription (e.g., 'en', 'auto'). 
                  Note: ggml-base.en.bin is English-only.

    Returns:
        Dictionary with 'text' (full transcript) and 'segments' (list of timestamped parts).
    """
    if not os.path.isfile(WHISPER_CPP_MAIN_EXE):
        logger.error(f"Whisper.cpp main.exe not found at: {WHISPER_CPP_MAIN_EXE}")
        raise FileNotFoundError(f"Whisper.cpp main.exe not found. Please ensure it's correctly set up.")
    if not os.path.isfile(WHISPER_CPP_MODEL_PATH):
        logger.error(f"Whisper.cpp model not found at: {WHISPER_CPP_MODEL_PATH}")
        raise FileNotFoundError(f"Whisper.cpp model not found. Please ensure it's correctly set up.")

    temp_file_path = ""
    try:
        # Create a temporary file for the audio data with .wav suffix
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        logger.info(f"Temporary audio file for whisper.cpp: {temp_file_path}")

        # Prepare the command for whisper.cpp main.exe
        cmd = [
            WHISPER_CPP_MAIN_EXE,
            "-m", WHISPER_CPP_MODEL_PATH,
            "-f", temp_file_path,
            "-l", language, # Model ggml-base.en.bin is English-only
            # "-t", "4",  # Example: Number of threads, adjust based on your CPU
            "--no-timestamps", "false", # Explicitly ask for timestamps (default behavior)
            # "--output-txt" # If specified without a file, prints to stdout.
                           # Default behavior is to print timestamped lines to stdout.
        ]
        
        logger.info(f"Executing whisper.cpp command: {' '.join(cmd)}")
        
        # subprocess.CREATE_NO_WINDOW is Windows-specific to hide the console window.
        creation_flags = 0
        if os.name == 'nt':
            creation_flags = subprocess.CREATE_NO_WINDOW

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            creationflags=creation_flags
        )
        
        # Timeout for the process to complete (e.g., 30 seconds for a chunk)
        stdout, stderr = process.communicate(timeout=30) 

        if process.returncode != 0:
            logger.error(f"whisper.cpp failed with exit code {process.returncode}")
            logger.error(f"whisper.cpp stderr: {stderr.strip()}")
            raise Exception(f"whisper.cpp transcription failed. Error: {stderr.strip()}")
            
        logger.info(f"whisper.cpp stdout raw: \n{stdout}")
        if stderr.strip(): # whisper.cpp often prints progress/model info to stderr
             logger.info(f"whisper.cpp stderr (info/progress): \n{stderr.strip()}")

        segments = parse_whisper_cpp_output(stdout)
        
        if not segments and stdout.strip(): # If parsing failed but there was output
            logger.warning("No segments parsed, but stdout was not empty. Raw output used as fallback.")
            # Fallback for non-timestamped or differently formatted output if parsing fails
            full_text = stdout.strip()
        else:
            full_text = " ".join([seg["text"] for seg in segments]).strip()

        return {
            "text": full_text,
            "segments": segments
        }
            
    except subprocess.TimeoutExpired:
        logger.error(f"whisper.cpp command timed out after 30s for file: {temp_file_path}")
        if process:
            process.kill()
            # Try to get any remaining output
            try:
                stdout, stderr = process.communicate(timeout=1)
                logger.error(f"Timeout stdout: {stdout.strip() if stdout else ''}")
                logger.error(f"Timeout stderr: {stderr.strip() if stderr else ''}")
            except Exception as e_comm:
                logger.error(f"Error getting output after timeout kill: {e_comm}")
        raise Exception("whisper.cpp transcription timed out.")
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found error during whisper.cpp transcription: {str(fnf_error)}")
        raise # Re-raise specific FileNotFoundError
    except Exception as e:
        logger.error(f"Generic error during whisper.cpp transcription for file {temp_file_path}: {str(e)}")
        # Log traceback for unexpected errors
        import traceback
        logger.error(traceback.format_exc())
        raise Exception(f"An unexpected error occurred during whisper.cpp transcription: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Successfully deleted temp file: {temp_file_path}")
            except Exception as e_del:
                logger.error(f"Error deleting temp file {temp_file_path}: {str(e_del)}")

