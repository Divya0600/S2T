import logging
import requests

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Ollama API configuration
OLLAMA_API_URL = "http://158.234.207.71:7861/api/generate"
OLLAMA_MODEL = "gemma3:4b"  # Default model

def generate_response(prompt, system_message=None, model=OLLAMA_MODEL, max_tokens=1500):
    """Send a prompt to Ollama's API and return the response text directly.
    
    Args:
        prompt: The main text prompt to process
        system_message: Optional system message to prepend to the prompt
        model: Model to use for inference
        max_tokens: Maximum number of tokens in the response
        
    Returns:
        str: The generated response text
    """
    try:
        # Combine system message and prompt if both are provided
        full_prompt = f"{system_message.strip()}\n\n{prompt}" if system_message else prompt
        
        logger.info(f"Sending request to Ollama API with model: {model}")
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "max_tokens": max_tokens
            },
            timeout=60  # Adding a timeout to prevent hanging
        )
        response.raise_for_status()
        data = response.json()
        result = data.get("response", "").strip()
        
        # If we get an empty result, log this and return a default message
        if not result or len(result.strip()) == 0:
            logger.warning("Received empty response from Ollama API")
            return "Unable to generate response. Please try again with a longer transcript."
            
        # Count tokens for logging purposes
        token_count = len(result.split())
        logger.info(f"Response received with approximately {token_count} words")
        
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Ollama API: {str(e)}")
        return f"Failed to communicate with Ollama service: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error during Ollama API call: {str(e)}")
        return f"Error: {str(e)}"
