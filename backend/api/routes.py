from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from time import time
import asyncio
import json
import os
import tempfile
import shutil
from typing import Optional, Union, List, Dict, Any
from api.ollama_handler import generate_response
from api.faster_transcription import FasterTranscriber, get_transcriber

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic in-memory rate limiting: allow up to 5 requests per minute per IP
RATE_LIMIT = 5
RATE_PERIOD = 60000  # seconds
ip_requests = {}

async def rate_limiter(request: Request):
    client_ip = request.client.host
    now = time()
    if client_ip not in ip_requests:
        ip_requests[client_ip] = []
    # Remove timestamps older than RATE_PERIOD
    ip_requests[client_ip] = [timestamp for timestamp in ip_requests[client_ip] if now - timestamp < RATE_PERIOD]
    if len(ip_requests[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests")
    ip_requests[client_ip].append(now)

class SummaryRequest(BaseModel):
    transcript: str

class ActionItemsRequest(BaseModel):
    transcript: str

import logging

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(levelname)s: %(message)s'  # Simplified format
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

@app.post("/summarize")
async def generate_summary(request: SummaryRequest, req: Request, limiter: None = Depends(rate_limiter)):
    """
    Generate a meeting summary from the provided transcript.
    """
    try:
        system_prompt = (
            "You are a professional meeting minutes writer.\n"
            "Your task is to generate a concise, well-structured summary of a meeting using only the information from the transcript provided.\n\n"
            "Please format the summary with these sections:\n"
            "1. Meeting Overview\n"
            "2. Key Discussion Points\n"
            "3. Action Items\n\n"
            "Guidelines:\n"
            "- Do NOT invent or assume any information not present in the transcript.\n"
            "- Be brief and to the point. Avoid repetition.\n"

            "Transcript:\n"
        )

        
        # Log the transcript length for debugging
        transcript_length = len(request.transcript)
        logger.warning(f"Received transcript with length: {transcript_length}")
        
        # Add a check for minimum transcript length
        if transcript_length < 20:  # Arbitrary minimum length
            logger.warning(f"Transcript too short: {request.transcript}")
            return {"status": "error", "detail": "Transcript too short to generate meaningful summary"}
            
        # Directly call Ollama API and get the response
        summary = generate_response(prompt=request.transcript, system_message=system_prompt, max_tokens=1500)
        return {"status": "completed", "summary": summary}
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-action-items")
async def extract_action_items(request: Union[ActionItemsRequest, dict], req: Request, limiter: None = Depends(rate_limiter)):
    """
    Extract action items from the meeting transcript.
    
    Expected request format:
    {
        "transcript": "[meeting transcript text]"
    }
    """
    try:
        # Log the incoming request for debugging
        logger.info(f"Received extract_action_items request")
        
        # Handle both Pydantic model and dict formats
        if isinstance(request, dict):
            if 'transcript' not in request:
                logger.error(f"Invalid request format: {request}")
                raise HTTPException(
                    status_code=422,
                    detail="Invalid request format. Expected {'transcript': 'your_transcript_text_here'}"
                )
            transcript = request.get('transcript', '').strip()
        else:
            transcript = request.transcript.strip()
        
        # Validate transcript content
        if not transcript:
            logger.error("Empty transcript provided")
            return {
                "status": "error", 
                "detail": "Empty transcript provided"
            }
            
        system_prompt = (
            "Extract action items from the following meeting transcript. For each action item, identify:\n"
            "1. The specific task to be done\n"
            "2. The person or team assigned to it\n\n"
            "Format the response as a clear list of action items. If no clear action items can be identified, \n"
            "state that no specific action items were found.\n\n"
            "Transcript:\n"
        )
        
        # Log the transcript length for debugging
        transcript_length = len(transcript)
        logger.info(f"Processing transcript with length: {transcript_length} characters")
        
        # Check for minimum transcript length
        if transcript_length < 10:  # Arbitrary minimum length
            logger.warning("Transcript too short for action item extraction")
            return {
                "status": "completed", 
                "action_items": "Transcript too short to extract meaningful action items"
            }
            
        try:
            # Call Ollama API and get the response
            logger.info("Sending request to Ollama API for action item extraction")
            response = generate_response(
                prompt=transcript, 
                system_message=system_prompt, 
                max_tokens=1000
            )
            
            # Log successful response
            logger.info("Successfully extracted action items")
            return {
                "status": "completed", 
                "action_items": response
            }
            
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error processing action items: {str(e)}"
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in extract_action_items: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="An unexpected error occurred while processing your request"
        )


# File upload and transcription endpoints

@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...), engine: str = Form("faster_whisper")):
    temp_dir = None
    temp_file_path = None
    
    try:
        start_time = time()
        logger.info(f"Starting transcription for file: {file.filename}, content_type: {file.content_type}")
        
        # Create a temporary directory for the file
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename or "audio.wav")
        
        logger.info(f"Saving uploaded file to temporary location: {temp_file_path}")
        
        # Save uploaded file to temp location
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Verify the file exists and has content
        if not os.path.exists(temp_file_path):
            raise Exception(f"Failed to save file to {temp_file_path}")
            
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"File saved successfully. Size: {file_size} bytes")
        
        if file_size == 0:
            raise Exception("Uploaded file is empty")
        
        # Initialize the transcriber with explicit parameters
        logger.info("Initializing FasterTranscriber...")
        transcriber = FasterTranscriber(
            model_name="base",
            language="en",
            device_type="cpu"
        )
        
        logger.info("Starting transcription...")
        result = await transcriber.transcribe_file(temp_file_path)
        
        # Log result format and content
        if isinstance(result, dict):
            text_length = len(result.get('text', ''))
            segments_count = len(result.get('segments', []))
            logger.info(f"Transcription completed. Transcript length: {text_length}, Segments: {segments_count}")
            
            # Return both text and segments for the frontend
            return {
                "status": "completed", 
                "transcript": result.get('text', ''),
                "segments": result.get('segments', [])
            }
        else:
            # Handle legacy format (string only)
            logger.info(f"Transcription completed with legacy format. Transcript length: {len(result) if result else 0}")
            return {"status": "completed", "transcript": result}
            
    except Exception as e:
        logger.error(f"Error transcribing file: {str(e)}", exc_info=True)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error transcribing file: {str(e)}"
        )
        
    finally:
        # Clean up temporary files
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.error(f"Error deleting temp file {temp_file_path}: {e}")
                
        if temp_dir and os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                logger.error(f"Error deleting temp directory {temp_dir}: {e}")

@app.get("/engines")
async def list_engines():
    """
    List available transcription engines
    """
    try:
        # Check if faster-whisper is available (requires GPU and proper setup)
        try:
            transcriber = FasterTranscriber()
            faster_whisper_available = True
        except Exception as e:
            logger.error(f"Error checking faster-whisper availability: {e}")
            faster_whisper_available = False

        engines = [
            {
                "id": "faster_whisper",
                "name": "FasterWhisper (GPU)",
                "available": faster_whisper_available,
                "description": "Server-based transcription using faster-whisper. Requires GPU."
            }
        ]
        return {
            "engines": engines,
            "default": "faster_whisper"
        }
    except Exception as e:
        logger.error(f"Error listing engines: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Real-time transcription endpoints

class TranscriberConfig(BaseModel):
    model_name: str = "base"
    language: Optional[str] = None
    chunk_duration_ms: int = 5000
    engine: str = "whisper"  # Options: "faster_whisper", "xenova", or "whisper"

@app.post("/transcriber/start")
async def start_transcription(config: TranscriberConfig):
    """Start the real-time transcription service with the given configuration."""
    try:
        # Convert model to configuration dictionary
        transcriber_config = config.dict()
        
        # Log the engine choice
        engine = transcriber_config.get('engine', 'whisper')
        logger.info(f"Starting transcription with engine: {engine}")
        
        # If using whisper, initialize the model but don't start anything
        # (will be handled by the WebSocket connection)
        if engine == 'whisper':
            model_name = transcriber_config.get('model_name', 'base')
            # Just initialize the model to make sure it loads
            transcriber = get_whisper_transcriber(model_name)
            return {"status": "ready", "config": transcriber_config}
        else:
            # Get or create the transcriber instance for faster-whisper or xenova
            transcriber = get_transcriber(config=transcriber_config)
            
            # Start the transcription
            transcriber.start()
            
            return {"status": "started", "config": transcriber_config}
    except Exception as e:
        logger.error(f"Error starting transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcriber/stop")
async def stop_transcription():
    """Stop the real-time transcription service."""
    try:
        transcriber = get_transcriber()
        if transcriber:
            transcriber.stop()
            return {"status": "stopped"}
        else:
            return {"status": "not_running"}
    except Exception as e:
        logger.error(f"Error stopping transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe-live-chunk")
async def transcribe_live_audio_chunk_endpoint(file: UploadFile = File(...)):
    """
    Receive an audio chunk (expected WAV format), transcribe it using faster-whisper,
    and return timestamped segments.
    """
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="No audio data received.")

        logger.info(f"Received live audio chunk, size: {len(audio_bytes)} bytes")
        
        # Create a temporary file to store the audio chunk
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Get the faster-whisper transcriber
            transcriber = get_transcriber()
            
            # Transcribe the audio chunk
            segments, info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: transcriber.model.transcribe(
                    temp_file_path,
                    language="en",
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
            )
            
            # Convert segments to the expected format
            result = {
                "text": " ".join(segment.text for segment in segments),
                "segments": [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    }
                    for segment in segments
                ]
            }
            
            logger.info(f"Live chunk transcription completed. Text length: {len(result['text'])}")
            return result
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.error(f"Error cleaning up temp file {temp_file_path}: {e}")

    except Exception as e:
        logger.error(f"Error processing live audio chunk: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error transcribing live audio chunk: {str(e)}")

@app.websocket("/transcriber/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time transcription results."""
    await websocket.accept()
    
    try:
        # Get config from first message
        config_msg = await websocket.receive_text()
        config = json.loads(config_msg)
        
        # Force faster-whisper for live transcription
        config['engine'] = 'faster_whisper'
        
        # Initialize transcriber with faster-whisper
        transcriber = get_transcriber(config=config)
        transcriber.start()
        
        # Process incoming audio chunks if needed or use internal mic capture
        # ...
        
        # Send transcription results back
        while True:
            result = transcriber.get_latest_result(block=False)
            if result:
                # Format result for frontend
                formatted_result = {
                    "text": result.get("text", ""),
                    "segments": [
                        {"text": seg["text"], "start": seg.get("start"), "end": seg.get("end")} 
                        for seg in result.get("segments", [])
                    ],
                    "timestamp": time()
                }
                await websocket.send_text(json.dumps(formatted_result))
            
            await asyncio.sleep(0.1)  # Poll every 100ms
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        # Cleanup transcription
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
            await websocket.close()
        except:
            pass
