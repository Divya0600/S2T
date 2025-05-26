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
import wave
import io
import scipy.signal

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RATE_LIMIT = 5
RATE_PERIOD = 60000
ip_requests = {}

async def rate_limiter(request: Request):
    client_ip = request.client.host
    now = time()
    if client_ip not in ip_requests:
        ip_requests[client_ip] = []
    ip_requests[client_ip] = [timestamp for timestamp in ip_requests[client_ip] if now - timestamp < RATE_PERIOD]
    if len(ip_requests[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests")
    ip_requests[client_ip].append(now)

class SummaryRequest(BaseModel):
    transcript: str

class ActionItemsRequest(BaseModel):
    transcript: str

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

live_transcription_sessions = {}

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

        
        transcript_length = len(request.transcript)
        logger.warning(f"Received transcript with length: {transcript_length}")
        
        if transcript_length < 20:
            logger.warning(f"Transcript too short: {request.transcript}")
            return {"status": "error", "detail": "Transcript too short to generate meaningful summary"}
            
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
        logger.info(f"Received extract_action_items request")
        
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
        
        transcript_length = len(transcript)
        logger.info(f"Processing transcript with length: {transcript_length} characters")
        
        if transcript_length < 10:
            logger.warning("Transcript too short for action item extraction")
            return {
                "status": "completed", 
                "action_items": "Transcript too short to extract meaningful action items"
            }
            
        try:
            logger.info("Sending request to Ollama API for action item extraction")
            response = generate_response(
                prompt=transcript, 
                system_message=system_prompt, 
                max_tokens=1000
            )
            
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
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in extract_action_items: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="An unexpected error occurred while processing your request"
        )


@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...), engine: str = Form("faster_whisper")):
    temp_dir = None
    temp_file_path = None
    
    try:
        start_time = time()
        logger.info(f"Starting transcription for file: {file.filename}, content_type: {file.content_type}")
        
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename or "audio.wav")
        
        logger.info(f"Saving uploaded file to temporary location: {temp_file_path}")
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if not os.path.exists(temp_file_path):
            raise Exception(f"Failed to save file to {temp_file_path}")
            
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"File saved successfully. Size: {file_size} bytes")
        
        if file_size == 0:
            raise Exception("Uploaded file is empty")
        
        logger.info("Initializing FasterTranscriber...")
        transcriber = FasterTranscriber(
            model_name="base",
            language="en",
            device_type="cpu"
        )
        
        logger.info("Starting transcription...")
        result = await transcriber.transcribe_file(temp_file_path)
        
        if isinstance(result, dict):
            text_length = len(result.get('text', ''))
            segments_count = len(result.get('segments', []))
            logger.info(f"Transcription completed. Transcript length: {text_length}, Segments: {segments_count}")
            
            return {
                "status": "completed", 
                "transcript": result.get('text', ''),
                "chunks": result.get('segments', [])
            }
        else:
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

class TranscriberConfig(BaseModel):
    model_name: str = "base"
    language: Optional[str] = None
    chunk_duration_ms: int = 3000
    engine: str = "faster_whisper"

@app.post("/transcriber/start")
async def start_transcription(config: TranscriberConfig):
    """Start the real-time transcription service with the given configuration."""
    try:
        transcriber_config = config.dict()
        transcriber_config.pop('engine', None)
        
        engine = config.engine
        logger.info(f"Starting transcription with engine: {engine}")
        
        transcriber = get_transcriber(config=transcriber_config)
        
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

def apply_noise_reduction(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Apply advanced noise reduction to audio data."""
    try:
        if len(audio_data) < 1024:
            return audio_data
        
        audio_float = audio_data.astype(np.float32)
        
        if np.max(np.abs(audio_float)) > 0:
            audio_float = audio_float / np.max(np.abs(audio_float))
        
        b, a = scipy.signal.butter(4, [80, 8000], btype='band', fs=sample_rate)
        filtered_audio = scipy.signal.filtfilt(b, a, audio_float)
        
        window_size = min(256, len(filtered_audio) // 4)
        if window_size > 0:
            noise_profile = np.median(np.abs(filtered_audio[:window_size]))
            filtered_audio = np.where(
                np.abs(filtered_audio) > noise_profile * 2.5,
                filtered_audio,
                filtered_audio * 0.1
            )
        
        return filtered_audio
        
    except Exception as e:
        logger.error(f"Error in noise reduction: {e}")
        return audio_data

async def process_audio_chunk_enhanced(session_state: dict, audio_data: np.ndarray, sample_rate: int = 16000):
    """Enhanced audio chunk processing with better error handling."""
    try:
        transcriber = session_state["transcriber"]
        
        if not transcriber or not transcriber.model:
            logger.error("Transcriber not initialized")
            return None
        
        audio_level = np.sqrt(np.mean(audio_data ** 2))
        logger.debug(f"Audio level: {audio_level:.6f}")
        
        if audio_level < 0.001:
            return None
        
        audio_data = apply_noise_reduction(audio_data, sample_rate)
        
        segments, info = transcriber.model.transcribe(
            audio_data,
            language="en",
            beam_size=5,
            best_of=5,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.3,
                min_speech_duration_ms=100,
                max_speech_duration_s=float('inf'),
                min_silence_duration_ms=300,
                window_size_samples=512,
                speech_pad_ms=200
            ),
            without_timestamps=False,
            word_timestamps=False,
            initial_prompt=None,
            suppress_blank=True,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.5,
            condition_on_previous_text=False
        )
        
        return list(segments), info
        
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}", exc_info=True)
        return None

async def handle_audio_data_enhanced(session_state: dict, data: dict):
    """Enhanced audio data handler with improved buffering and processing."""
    try:
        audio_array = data.get("audio", [])
        sample_rate = data.get("sample_rate", 16000)
        timestamp = data.get("timestamp", time())
        
        if not audio_array:
            return
        
        audio_np = np.array(audio_array, dtype=np.float32)
        
        max_val = np.abs(audio_np).max()
        if max_val > 0:
            audio_np = audio_np / max_val
        
        session_state["audio_buffer"].append({
            "data": audio_np,
            "timestamp": timestamp
        })
        
        total_samples = sum(len(chunk["data"]) for chunk in session_state["audio_buffer"])
        total_duration = total_samples / sample_rate
        
        if total_duration >= 1.0:
            combined_audio = np.concatenate([chunk["data"] for chunk in session_state["audio_buffer"]])
            
            chunk_timestamp = session_state["audio_buffer"][0]["timestamp"]
            
            result = await process_audio_chunk_enhanced(session_state, combined_audio, sample_rate)
            
            if result and result[0]:
                segments_list, info = result
                new_segments = []
                
                base_time = (chunk_timestamp - session_state["start_time"]) / 1000.0
                
                for segment in segments_list:
                    segment_text = segment.text.strip()
                    
                    if len(segment_text) < 2:
                        continue
                    
                    hallucinations = [
                        "thanks", "thank you", "thanks for watching",
                        "please subscribe", "like and subscribe",
                        "bye", "goodbye", "[music]", "[applause]",
                        "♪", "um", "uh", "hmm", "mm"
                    ]
                    
                    if any(hall in segment_text.lower() for hall in hallucinations):
                        continue
                    
                    if len(segment_text.split()) < 2:
                        continue
                    
                    segment_start = base_time + segment.start
                    segment_end = base_time + segment.end
                    
                    new_segments.append({
                        "text": segment_text,
                        "start": segment_start,
                        "end": segment_end
                    })
                
                if new_segments:
                    session_state["current_segments"].extend(new_segments)
                    all_text = " ".join(seg["text"] for seg in session_state["current_segments"])
                    session_state["current_text"] = all_text
                    
                    result = {
                        "text": all_text,
                        "segments": new_segments,
                        "timestamp": timestamp,
                        "is_partial": True,
                        "session_id": session_state.get("session_id", "unknown")
                    }
                    
                    try:
                        await session_state["websocket"].send_text(json.dumps(result))
                        logger.info(f"Sent {len(new_segments)} segments to client")
                    except Exception as e:
                        logger.error(f"Error sending result: {e}")
            
            overlap_samples = int(0.25 * sample_rate)
            if len(combined_audio) > overlap_samples:
                overlap_audio = combined_audio[-overlap_samples:]
                session_state["audio_buffer"] = [{
                    "data": overlap_audio,
                    "timestamp": timestamp
                }]
            else:
                session_state["audio_buffer"] = []
            
    except Exception as e:
        logger.error(f"Error handling audio data: {e}", exc_info=True)

@app.websocket("/transcriber/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint for real-time transcription."""
    session_id = f"session_{int(time() * 1000)}"
    
    await websocket.accept()
    logger.info(f"WebSocket connection established: {session_id}")
    
    session_state = {
        "is_active": True,
        "transcriber": None,
        "current_text": "",
        "current_segments": [],
        "websocket": websocket,
        "audio_buffer": [],
        "start_time": time() * 1000,
        "session_id": session_id,
        "last_activity": time(),
        "silence_duration": 0,
        "last_audio_time": time()
    }
    
    live_transcription_sessions[session_id] = session_state
    
    try:
        query_params = dict(websocket.query_params)
        engine = query_params.get('engine', 'faster_whisper')
        
        logger.info(f"Initializing session {session_id} with engine: {engine}")
        
        config = {
            'model_name': 'base',
            'language': 'en',
            'chunk_duration_ms': 1000,
            'compute_type': 'int8',
            'device_type': 'cpu'
        }
        
        transcriber = FasterTranscriber(**config)
        session_state["transcriber"] = transcriber
        
        logger.info(f"Live transcription ready for session {session_id}")
        
        while session_state["is_active"]:
            try:
                try:
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                    session_state["last_activity"] = time()
                    
                    data = json.loads(message)
                    
                    if data.get("type") == "audio_data":
                        session_state["last_audio_time"] = time()
                        await handle_audio_data_enhanced(session_state, data)
                    elif data.get("command") == "stop":
                        logger.info(f"Stop command received for session {session_id}")
                        break
                    else:
                        logger.debug(f"Configuration update: {data}")
                        
                except asyncio.TimeoutError:
                    current_time = time()
                    
                    if current_time - session_state["last_audio_time"] > 5:
                        session_state["audio_buffer"] = []
                    
                    if current_time - session_state["last_activity"] > 15:
                        try:
                            await websocket.send_text(json.dumps({
                                "heartbeat": True,
                                "session_id": session_id,
                                "timestamp": current_time
                            }))
                        except:
                            logger.info(f"Client disconnected: {session_id}")
                            break
                    continue
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {session_id}")
                break
            except Exception as e:
                logger.error(f"Error in message loop for {session_id}: {e}")
                await asyncio.sleep(0.1)
                
    except Exception as e:
        logger.error(f"Error in WebSocket session {session_id}: {e}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({
                "error": str(e),
                "session_id": session_id
            }))
        except:
            pass
    finally:
        logger.info(f"Cleaning up session: {session_id}")
        
        if session_id in live_transcription_sessions:
            session_state = live_transcription_sessions[session_id]
            session_state["is_active"] = False
            
            transcriber = session_state.get("transcriber")
            if transcriber and hasattr(transcriber, 'stop'):
                try:
                    transcriber.stop()
                except:
                    pass
            
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
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        try:
            transcriber = get_transcriber()
            
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
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.error(f"Error cleaning up temp file {temp_file_path}: {e}")

    except Exception as e:
        logger.error(f"Error processing live audio chunk: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error transcribing live audio chunk: {str(e)}")