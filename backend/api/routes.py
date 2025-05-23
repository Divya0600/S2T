from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from time import time
import asyncio
import json
import os
import tempfile
import shutil
import numpy as np
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

# Global state for live transcription - IMPROVED
live_transcription_sessions = {}  # Store sessions by WebSocket ID

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
                "chunks": result.get('segments', [])  # Use 'chunks' key for consistency with frontend
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
                "name": "FasterWhisper (CPU/GPU)",
                "available": faster_whisper_available,
                "description": "Server-based transcription using faster-whisper with improved real-time processing."
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
    chunk_duration_ms: int = 3000  # Reduced for better real-time performance
    engine: str = "faster_whisper"

@app.post("/transcriber/start")
async def start_transcription(config: TranscriberConfig):
    """Start the real-time transcription service with the given configuration."""
    try:
        # Convert model to configuration dictionary and remove engine parameter
        transcriber_config = config.dict()
        transcriber_config.pop('engine', None)  # Remove engine as it's not a FasterTranscriber parameter
        
        # Log the engine choice
        engine = config.engine
        logger.info(f"Starting transcription with engine: {engine}")
        
        # Get or create the transcriber instance
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

async def handle_audio_data(session_state: dict, data: dict):
    """Handle incoming audio data and perform transcription."""
    try:
        audio_array = data.get("audio", [])
        sample_rate = data.get("sample_rate", 16000)
        timestamp = data.get("timestamp", time())
        
        if not audio_array:
            return
        
        # Convert to numpy array
        audio_np = np.array(audio_array, dtype=np.float32)
        
        # Add to buffer
        session_state["audio_buffer"].append(audio_np)
        
        # Process when we have enough audio (about 2.5 seconds)
        total_duration = sum(len(chunk) for chunk in session_state["audio_buffer"]) / sample_rate
        
        if total_duration >= 2.5:  # Process every 2.5 seconds for good real-time response
            # Combine all audio chunks
            combined_audio = np.concatenate(session_state["audio_buffer"])
            
            # Skip if audio is too quiet (likely silence or noise)
            audio_level = np.sqrt(np.mean(combined_audio ** 2))
            
            if audio_level > 0.005:  # Threshold to avoid processing silence
                try:
                    # Get transcriber
                    transcriber = session_state["transcriber"]
                    
                    if transcriber and transcriber.model:
                        # Transcribe the audio chunk
                        segments, info = transcriber.model.transcribe(
                            combined_audio,
                            language="en",
                            beam_size=3,  # Fast processing
                            vad_filter=True,
                            vad_parameters=dict(
                                min_silence_duration_ms=500,
                                speech_pad_ms=200
                            )
                        )
                        
                        # Process segments
                        new_segments = []
                        current_time = time() - session_state["start_time"]
                        
                        for segment in segments:
                            segment_text = segment.text.strip()
                            
                            # Skip very short or empty segments
                            if len(segment_text) < 3:
                                continue
                            
                            # Skip common hallucinations that Whisper generates
                            hallucinations = [
                                "thanks for watching", "thank you for watching", 
                                "bye", "goodbye", "see you later",
                                "subscribe", "like and subscribe",
                                "thanks", "thank you", "music",
                                "applause", "[music]", "[applause]"
                            ]
                            
                            if segment_text.lower() in hallucinations:
                                logger.debug(f"Skipping likely hallucination: {segment_text}")
                                continue
                            
                            # Create segment with proper timestamps
                            segment_start = current_time + segment.start
                            segment_end = current_time + segment.end
                            
                            segment_data = {
                                "text": segment_text,
                                "start": segment_start,
                                "end": segment_end
                            }
                            
                            new_segments.append(segment_data)
                        
                        # Send results if we have new segments
                        if new_segments:
                            # Update session state
                            session_state["current_segments"].extend(new_segments)
                            all_text = " ".join(seg["text"] for seg in session_state["current_segments"])
                            session_state["current_text"] = all_text
                            
                            # Send to client
                            result = {
                                "text": all_text,
                                "segments": new_segments,  # Only new segments
                                "timestamp": timestamp,
                                "is_partial": True,
                                "session_id": session_state.get("session_id", "unknown")
                            }
                            
                            try:
                                await session_state["websocket"].send_text(json.dumps(result))
                                logger.info(f"Sent {len(new_segments)} new segments to client")
                            except Exception as e:
                                logger.error(f"Error sending transcription result: {e}")
                        else:
                            logger.debug("No valid segments found in audio chunk")
                            
                except Exception as e:
                    logger.error(f"Error transcribing audio chunk: {e}")
            else:
                logger.debug(f"Audio too quiet (level: {audio_level:.6f}), skipping transcription")
            
            # Clear buffer after processing
            session_state["audio_buffer"] = []
            
    except Exception as e:
        logger.error(f"Error handling audio data: {e}")

@app.websocket("/transcriber/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint for real-time transcription with direct audio streaming."""
    session_id = f"session_{int(time() * 1000)}"
    
    await websocket.accept()
    logger.info(f"WebSocket connection established: {session_id}")
    
    # Initialize session state
    session_state = {
        "is_active": True,
        "transcriber": None,
        "processed_segments": set(),
        "current_text": "",
        "current_segments": [],
        "websocket": websocket,
        "last_chunk_id": 0,
        "audio_buffer": [],
        "start_time": time(),
        "session_id": session_id
    }
    
    live_transcription_sessions[session_id] = session_state
    
    try:
        # Parse query parameters
        query_params = dict(websocket.query_params)
        engine = query_params.get('engine', 'faster_whisper')
        
        logger.info(f"Initializing session {session_id} with engine: {engine}")
        
        # Initialize transcriber with settings optimized for live transcription
        config = {
            'model_name': 'base',
            'language': 'en',
            'chunk_duration_ms': 1500,  # Shorter chunks for better real-time response
            'compute_type': 'int8',
            'device_type': 'cpu'
        }
        
        transcriber = FasterTranscriber(**config)
        session_state["transcriber"] = transcriber
        
        logger.info(f"Real-time transcription initialized for session {session_id}")
        
        # Main message handling loop
        while session_state["is_active"]:
            try:
                # Wait for message with timeout
                try:
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    data = json.loads(message)
                    
                    if data.get("type") == "audio_data":
                        # Handle incoming audio data for live transcription
                        await handle_audio_data(session_state, data)
                    elif data.get("command") == "stop":
                        logger.info(f"Received stop command for session {session_id}")
                        break
                    else:
                        logger.debug(f"Received configuration update: {data}")
                        
                except asyncio.TimeoutError:
                    # Send heartbeat if no messages received
                    if session_state["is_active"]:
                        try:
                            await websocket.send_text(json.dumps({
                                "heartbeat": True,
                                "session_id": session_id,
                                "timestamp": time()
                            }))
                        except:
                            logger.info(f"Failed to send heartbeat, connection likely closed: {session_id}")
                            break
                
            except Exception as e:
                logger.error(f"Error processing message for session {session_id}: {e}")
                await asyncio.sleep(0.1)
                
    except Exception as e:
        logger.error(f"Error in WebSocket connection {session_id}: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e), "session_id": session_id}))
        except:
            pass
    finally:
        # Cleanup session
        logger.info(f"Cleaning up WebSocket session: {session_id}")
        
        if session_id in live_transcription_sessions:
            session_state = live_transcription_sessions[session_id]
            session_state["is_active"] = False
            
            # Stop transcriber if it exists
            transcriber = session_state.get("transcriber")
            if transcriber:
                try:
                    transcriber.stop()
                    logger.info(f"Transcriber stopped for session {session_id}")
                except Exception as e:
                    logger.error(f"Error stopping transcriber for session {session_id}: {e}")
            
            # Remove session from global dict
            del live_transcription_sessions[session_id]
        
        try:
            await websocket.close()
        except:
            pass
        
        logger.info(f"Session {session_id} cleanup completed")

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