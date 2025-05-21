from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from time import time
import asyncio
import json
import os
import tempfile
import shutil
from typing import Optional, Union, List
from api.ollama_handler import generate_response
from api.faster_transcription import FasterTranscriber, get_transcriber
from api.whisper_handler import get_whisper_transcriber, transcribe_audio_chunk, transcribe_audio_chunk_cpp

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
            "You are a professional meeting minutes writer. Create a concise and structured Minutes of Meeting (MOM) with these sections:\n\n"
            "1. Meeting Overview\n"
            "2. Key Discussion Points\n"
            "3. Action Items\n\n"
            "IMPORTANT: Base your summary ONLY on the actual transcript provided below. Do NOT use generic content, placeholder names, or made-up information. If the transcript is unclear or too short, state that in your summary.\n\n"
            "Format your response in rich HTML format with proper headings (h2, h3), paragraphs, and formatting. Use <b> for important points, <ul> and <li> for lists where appropriate.\n\n"
            "Here's the transcript:\n"
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

@app.post("/extract-action-items")
async def extract_action_items(request: SummaryRequest, req: Request, limiter: None = Depends(rate_limiter)):
    """
    Extract action items from the meeting transcript.
    """
    try:
        system_prompt = (
            "Extract action items from the following meeting transcript. For each action item, identify:\n"
            "1. The specific task to be done\n"
            "2. The person or team assigned to it\n\n"
            "IMPORTANT: Base your extraction ONLY on the actual transcript provided below. Do NOT use generic content, placeholder names, or made-up information. If no clear action items or assignees are found, state that explicitly.\n\n"
            "Format your response as an HTML list with each action item formatted like this:\n"
            "<div class=\"action-item\">\n"
            "  <p class=\"task\"><b>Task:</b> [task description]</p>\n"
            "  <p class=\"assignee\"><b>Assignee:</b> [person/team]</p>\n"
            "</div>\n\n"
            "Here's the transcript:\n"
        )
        
        # Log the transcript length for debugging
        transcript_length = len(request.transcript)
        logger.warning(f"Received transcript for action items with length: {transcript_length}")
        
        # Add a check for minimum transcript length
        if transcript_length < 20:  # Arbitrary minimum length
            logger.warning(f"Transcript too short: {request.transcript}")
            return {"status": "error", "detail": "Transcript too short to extract meaningful action items"}
            
        # Directly call Ollama API and get the response
        response = generate_response(prompt=request.transcript, system_message=system_prompt, max_tokens=1000)
        return {"status": "completed", "action_items": response}
    except Exception as e:
        logger.error(f"Error extracting action items: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# File upload and transcription endpoints

@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...), engine: str = Form("faster_whisper")):
    try:
        start_time = time()
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        # Save uploaded file to temp location
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Use faster-whisper (server-based) transcription
        transcriber = FasterTranscriber()
        result = await transcriber.transcribe_file(temp_file_path)
        
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        
        logger.info(f"Transcription completed. Transcript length: {len(result)}")
        return {"status": "completed", "transcript": result}
            
    except Exception as e:
        logger.error(f"Error transcribing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    Receive an audio chunk (expected WAV format), transcribe it using whisper.cpp,
    and return timestamped segments.
    """
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="No audio data received.")

        logger.info(f"Received live audio chunk for whisper.cpp, size: {len(audio_bytes)} bytes")
        
        # Using the new whisper.cpp based transcriber
        # The ggml-base.en.bin model is English-only, so language='en' is appropriate.
        result = transcribe_audio_chunk_cpp(audio_data=audio_bytes, language="en")
        
        logger.info(f"whisper.cpp live chunk transcription result: {result}")
        return result

    except FileNotFoundError as e_fnf:
        logger.error(f"transcribe_live_audio_chunk_endpoint: Whisper.cpp file not found - {str(e_fnf)}")
        raise HTTPException(status_code=500, detail=f"Live transcription engine misconfiguration: {str(e_fnf)}")
    except Exception as e:
        logger.error(f"Error processing live audio chunk with whisper.cpp: {str(e)}")
        # Consider logging traceback for detailed debugging
        # import traceback
        # logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error transcribing live audio chunk: {str(e)}")

@app.websocket("/transcriber/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time transcription results."""
    await websocket.accept()
    
    try:
        # Get config from first message
        config_msg = await websocket.receive_text()
        config = json.loads(config_msg)
        engine = config.get('engine', 'whisper')
        model_name = config.get('model_name', 'base')
        
        # For Whisper timestamps mode, we'll handle audio chunks directly
        if engine == 'whisper':
            # Initialize whisper transcriber
            transcriber = get_whisper_transcriber(model_name)
            
            # Process audio chunks and return transcripts with timestamps
            while True:
                # Receive audio chunk from WebSocket
                audio_chunk = await websocket.receive_bytes()
                
                # No need to process empty chunks
                if not audio_chunk:
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    # Process audio chunk with whisper
                    result = transcribe_audio_chunk(
                        audio_chunk, 
                        model_name=model_name,
                        language=config.get('language')
                    )
                    
                    # Send the transcript with timestamps back to the client
                    await websocket.send_text(json.dumps({
                        "text": result.get("text", ""),
                        "segments": result.get("segments", []),
                        "timestamp": time()
                    }))
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}")
                    await websocket.send_text(json.dumps({"error": str(e)}))
        else:
            # Use existing transcriber (faster-whisper) for non-timestamp mode
            transcriber = get_transcriber()
            if not transcriber:
                await websocket.send_text(json.dumps({
                    "error": "Transcriber not initialized. Call /transcriber/start first."
                }))
                await websocket.close()
                return
            
            # This will run until the WebSocket disconnects
            while True:
                # Check for new transcription results
                result = transcriber.get_latest_result(block=False)
                if result:
                    # Send the result to the WebSocket client
                    await websocket.send_text(json.dumps({
                        "text": result.get("text", ""),
                        "segments": result.get("segments", []),
                        "timestamp": time()
                    }))
                
                # Poll for results every 100ms
                await asyncio.sleep(0.1)
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
            await websocket.close()
        except:
            pass
