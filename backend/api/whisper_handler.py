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
