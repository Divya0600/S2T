"""
Xenova Whisper Implementation for Speech-to-Text

This module provides integration with the Xenova implementation of OpenAI's Whisper model
for speech-to-text transcription as an alternative to faster-whisper.
"""

import logging
import torch
from typing import Dict, Union, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flag to track if the Xenova modules have been imported
_xenova_imported = False
_transformers = None
_pipeline = None

def _lazy_import_xenova():
    """Lazily import Xenova modules to avoid loading them unnecessarily"""
    global _xenova_imported, _transformers, _pipeline
    
    if not _xenova_imported:
        try:
            # Import the required modules
            import transformers
            from transformers import pipeline
            
            _transformers = transformers
            _pipeline = pipeline
            _xenova_imported = True
            
            logger.info("Successfully imported Xenova modules")
        except ImportError:
            logger.error("Failed to import Xenova modules. Make sure transformers is installed.")
            raise

def transcribe_with_xenova(
    audio_path: str,
    model_name: str = "openai/whisper-small",
    language: Optional[str] = None,
    return_timestamps: bool = False
) -> Union[str, Dict[str, Any]]:
    """
    Transcribe audio using the Xenova implementation of Whisper.
    
    Args:
        audio_path: Path to the audio file
        model_name: Whisper model to use
        language: Optional language code (if None, language will be auto-detected)
        return_timestamps: Whether to return word-level timestamps
        
    Returns:
        Transcribed text or dictionary with text and metadata
    """
    # Lazy import Xenova modules
    _lazy_import_xenova()
    
    try:
        logger.info(f"Loading Xenova Whisper model: {model_name}")
        
        # Check for GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Initialize the pipeline
        pipe = _pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device
        )
        
        # Set additional parameters
        transcription_kwargs = {}
        if language:
            transcription_kwargs["language"] = language
        if return_timestamps:
            transcription_kwargs["return_timestamps"] = return_timestamps
            
        # Perform transcription
        logger.info(f"Starting transcription of {audio_path}")
        result = pipe(audio_path, **transcription_kwargs)
        
        # Process and return results
        if return_timestamps:
            return result
        else:
            # Return just the text for simple transcriptions
            return result.get("text", "")
        
    except Exception as e:
        logger.error(f"Error in Xenova transcription: {str(e)}")
        raise
