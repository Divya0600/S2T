import { useCallback, useMemo, useState, useEffect, useRef } from "react";
import { useWorker } from "./useWorker";
import Constants from "../utils/Constants";
import axios from "axios";

// Format seconds to HH:MM:SS format for timestamps
export const formatTimestamp = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    const pad = (num: number) => num.toString().padStart(2, '0');
    return `${pad(hours)}:${pad(minutes)}:${pad(secs)}`;
};

interface ProgressItem {
    file: string;
    loaded: number;
    progress: number;
    total: number;
    name: string;
    status: string;
}

interface TranscriberUpdateData {
    data: [
        string,
        { chunks: { text: string; timestamp: [number, number | null] }[] },
    ];
    text: string;
}

interface TranscriberCompleteData {
    data: {
        text: string;
        chunks: { text: string; timestamp: [number, number | null] }[];
    };
}

export interface TranscriptionSegment {
    text: string;
    start?: number;
    end?: number;
    timestamp?: [number, number | null];
}

export interface TranscriberData {
    isBusy: boolean;
    text: string;
    chunks: TranscriptionSegment[];
    segments?: TranscriptionSegment[];
}

// Define the return type for transcription results
interface TranscriptionResult {
    transcript?: string;
    summary?: string;
    error?: string;
    chunks?: { text: string; start?: number; end?: number; timestamp?: [number, number | null] }[];
}

export interface Transcriber {
    onInputChange: () => void;
    isBusy: boolean;
    isModelLoading: boolean;
    progressItems: ProgressItem[];
    start: (audioData: AudioBuffer | undefined) => void;
    output?: TranscriberData;
    
    // Model configuration
    model: string;
    setModel: (model: string) => void;
    multilingual: boolean;
    setMultilingual: (model: boolean) => void;
    quantized: boolean;
    setQuantized: (model: boolean) => void;
    subtask: string;
    setSubtask: (subtask: string) => void;
    language?: string;
    setLanguage: (language: string) => void;
    
    // Engine selection
    engine: string;
    setEngine: (engine: string) => void;
    
    // Live mode toggle
    isLiveMode: boolean;
    setIsLiveMode: (isLive: boolean) => void;
    
    // Recording status
    isRecording: boolean;
    recordedMedia: { url: string, type: string } | null;
    
    // File transcription
    transcribeFile: (file: File) => Promise<TranscriptionResult>;
    
    // Live streaming functionality
    startStreaming: (updateCallback: (text: string, segments?: Array<{text: string, start?: number, end?: number}>) => void, options?: {
        captureBothIO?: boolean;
        inputDevice?: string | number | null;
        outputDevice?: string | number | null;
    }) => void;
    stopStreaming: () => void;
    
    // Recording functionality
    startRecording: () => void;
    stopRecording: () => void;
    
    // Screen recording
    startScreenRecording: () => void;
    stopScreenRecording: () => void;
}

export function useTranscriber(): Transcriber {
    const [transcript, setTranscript] = useState<TranscriberData | undefined>(undefined);
    const [isBusy, setIsBusy] = useState(false);
    const [isModelLoading, setIsModelLoading] = useState(false);
    const [progressItems, setProgressItems] = useState<ProgressItem[]>([]);
    const [isRecording, setIsRecording] = useState(false);
    const [isLiveMode, setIsLiveMode] = useState(false);
    const [recordedMedia, setRecordedMedia] = useState<{ url: string, type: string } | null>(null);
    
    // Engine selection based on mode: faster-whisper for standard, whisper (OpenAI local) for live
    const [engine, setEngineState] = useState<string>("faster_whisper");
    
    // Update engine based on live mode
    useEffect(() => {
        const newEngine = isLiveMode ? "whisper" : "faster_whisper";
        console.log(`Setting engine to ${newEngine} based on isLiveMode=${isLiveMode}`);
        setEngineState(newEngine);
    }, [isLiveMode]);
    
    const setEngine = useCallback((newEngine: string) => {
        console.log(`Changing engine from ${engine} to ${newEngine}`);
        setEngineState(newEngine);
    }, [engine]);
    
    // Model configuration states
    const [model, setModel] = useState<string>(Constants.DEFAULT_MODEL);
    const [subtask, setSubtask] = useState<string>(Constants.DEFAULT_SUBTASK);
    const [quantized, setQuantized] = useState<boolean>(Constants.DEFAULT_QUANTIZED);
    const [multilingual, setMultilingual] = useState<boolean>(Constants.DEFAULT_MULTILINGUAL);
    const [language, setLanguage] = useState<string>(Constants.DEFAULT_LANGUAGE);
    
    // WebSocket reference for streaming audio
    const webSocketRef = useRef<WebSocket | null>(null);
    const isStreamingRef = useRef<boolean>(false);

    // Clean up audio resources when component unmounts
    useEffect(() => {
        return () => {
            if (webSocketRef.current) {
                webSocketRef.current.close();
            }
        };
    }, []);

    // Audio context and media recorder for live whisper transcription
    const audioContextRef = useRef<AudioContext | null>(null);
    const micStreamRef = useRef<MediaStream | null>(null);
    const streamingCallbackRef = useRef<((text: string, segments?: Array<{text: string, start?: number, end?: number}>) => void) | null>(null);
    
    // Media recorder for audio/screen recording
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const [error, setError] = useState<string>('');

    const webWorker = useWorker((event) => {
        const message = event.data;
        console.log('Received worker message:', message.status);
        
        // Update the state with the result
        switch (message.status) {
            case "progress":
                // Model file progress: update one of the progress items.
                console.log(`Loading model file: ${message.file}, Progress: ${message.progress}%`);
                setProgressItems((prev) =>
                    prev.map((item) => {
                        if (item.file === message.file) {
                            return { ...item, progress: message.progress };
                        }
                        return item;
                    }),
                );
                break;
            case "update":
                // Received partial update
                const updateMessage = message as TranscriberUpdateData;
                console.log('Received partial transcription update:', {
                    textLength: updateMessage.data[0].length,
                    chunksCount: updateMessage.data[1].chunks.length
                });
                setTranscript({
                    isBusy: true,
                    text: updateMessage.data[0],
                    chunks: updateMessage.data[1].chunks,
                });
                break;
            case "complete":
                // Received complete transcript
                const completeMessage = message as TranscriberCompleteData;
                console.log('Transcription completed:', {
                    textLength: completeMessage.data.text.length,
                    chunksCount: completeMessage.data.chunks.length
                });
                setTranscript({
                    isBusy: false,
                    text: completeMessage.data.text,
                    chunks: completeMessage.data.chunks,
                });
                setIsBusy(false);
                break;

            case "initiate":
                // Model file start load: add a new progress item to the list.
                console.log('Initiating model loading:', message);
                setIsModelLoading(true);
                setProgressItems((prev) => [...prev, message]);
                break;
            case "ready":
                console.log('Model is ready');
                setIsModelLoading(false);
                break;
            case "error":
                console.error('Transcription error:', message.data);
                setIsBusy(false);
                alert(
                    `${message.data.message} This is most likely because you are using Safari on an M1/M2 Mac. Please try again from Chrome, Firefox, or Edge.\n\nIf this is not the case, please file a bug report.`,
                );
                break;
            case "done":
                // Model file loaded: remove the progress item from the list.
                console.log(`Model file loaded: ${message.file}`);
                setProgressItems((prev) =>
                    prev.filter((item) => item.file !== message.file),
                );
                break;

            default:
                console.log('Unhandled message status:', message.status);
                break;
        }
    });

    const onInputChange = useCallback(() => {
        setTranscript(undefined);
    }, []);

    const postRequest = useCallback(
        async (audioData: AudioBuffer | undefined) => {
            if (audioData) {
                console.log('Starting audio transcription process');
                console.log('Audio details:', {
                    sampleRate: audioData.sampleRate,
                    numberOfChannels: audioData.numberOfChannels,
                    length: audioData.length,
                    duration: audioData.duration.toFixed(2) + ' seconds'
                });

                setTranscript(undefined);
                setIsBusy(true);

                let audio;
                if (audioData.numberOfChannels === 2) {
                    console.log('Processing stereo audio to mono');
                    const SCALING_FACTOR = Math.sqrt(2);

                    let left = audioData.getChannelData(0);
                    let right = audioData.getChannelData(1);

                    audio = new Float32Array(left.length);
                    for (let i = 0; i < audioData.length; ++i) {
                        audio[i] = SCALING_FACTOR * (left[i] + right[i]) / 2;
                    }
                } else {
                    console.log('Using mono audio directly');
                    audio = audioData.getChannelData(0);
                }

                console.log('Sending audio to worker with config:', {
                    model,
                    multilingual,
                    quantized,
                    subtask: multilingual ? subtask : null,
                    language: multilingual && language !== "auto" ? language : null
                });

                webWorker.postMessage({
                    audio,
                    model,
                    multilingual,
                    quantized,
                    subtask: multilingual ? subtask : null,
                    language:
                        multilingual && language !== "auto" ? language : null,
                });
            } else {
                console.warn('No audio data provided for transcription');
            }
        },
        [webWorker, model, multilingual, quantized, subtask, language],
    );

    // File transcription function
    const transcribeFile = useCallback(async (file: File): Promise<TranscriptionResult> => {
      setIsBusy(true);
      setTranscript({ text: '', chunks: [], isBusy: true }); // Reset transcript and indicate busy
      try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('engine', engine); // Use the current engine state
        
        const endpoint = 'http://localhost:8000/transcribe'; 
        
        console.log(`Transcribing file with ${engine} engine: ${file.name}`);
        
        const response = await axios.post(endpoint, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        
        if (response.data && response.data.transcript !== undefined) {
          const transcriptText = response.data.transcript || '';
          const responseChunks = response.data.chunks || [];
          
          setTranscript({
            isBusy: false,
            text: transcriptText,
            chunks: responseChunks,
          });
          
          return { transcript: transcriptText, chunks: responseChunks };
        } else {
          console.error('No data or transcript in response from transcription service:', response.data);
          setTranscript({ isBusy: false, text: '', chunks: [] });
          return { error: 'No transcript data received from transcription service' };
        }
      } catch (error) {
        console.error('Error transcribing file:', error);
        let errorMessage = 'Failed to transcribe file.';
        if (axios.isAxiosError(error) && error.response) {
          errorMessage = `Failed to transcribe file: ${error.response.status} ${error.response.data?.detail || error.message}`;
        } else if (error instanceof Error) {
          errorMessage = `Failed to transcribe file: ${error.message}`;
        }
        setTranscript({ isBusy: false, text: '', chunks: [] });
        return { error: errorMessage };
      } finally {
        setIsBusy(false);
      }
    }, [engine, setTranscript]);

    // Live streaming functionality - FIXED VERSION
    const startStreaming = useCallback(async (updateCallback: (text: string, segments?: Array<{text: string, start?: number, end?: number}>) => void, options?: {
        captureBothIO?: boolean;
        inputDevice?: string | number | null;
        outputDevice?: string | number | null;
    }) => {
        console.log('Starting streaming transcription with engine:', engine);
        
        if (isBusy || isStreamingRef.current) {
            console.log('Transcriber is busy or already streaming, cannot start streaming');
            return;
        }
        
        setIsBusy(true);
        isStreamingRef.current = true;
        streamingCallbackRef.current = updateCallback;
        
        // Clear any existing transcript when starting new stream
        setTranscript({
            isBusy: true,
            text: '',
            chunks: [],
        });
        
        // Close any existing WebSocket connection
        if (webSocketRef.current) {
            webSocketRef.current.close();
        }
        
        try {
            // Create a new WebSocket connection
            const wsUrl = `ws://localhost:8000/transcriber/ws?engine=faster-whisper`;
            webSocketRef.current = new WebSocket(wsUrl);
            
            webSocketRef.current.onopen = () => {
                console.log('WebSocket connection established');
                setIsBusy(false);
            };
            
            webSocketRef.current.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    // Only process if we're still streaming
                    if (!isStreamingRef.current) {
                        console.log('Ignoring message - streaming stopped');
                        return;
                    }
                    
                    if (data.text && streamingCallbackRef.current) {
                        // Process the segments to avoid duplicates
                        const segments = data.segments || [];
                        
                        // Update transcript state with the LATEST complete text
                        // Don't accumulate - just show the current complete transcription
                        setTranscript(prev => ({
                            isBusy: true,
                            text: data.text, // Use the complete text from the server
                            chunks: segments.map((seg: any) => ({
                                text: seg.text,
                                start: seg.start,
                                end: seg.end,
                                timestamp: [seg.start, seg.end]
                            })),
                        }));
                        
                        // Call the update callback with the latest text
                        streamingCallbackRef.current(data.text, segments);
                    }
                } catch (error) {
                    console.error('Error processing WebSocket message:', error);
                }
            };
            
            webSocketRef.current.onerror = (error) => {
                console.error('WebSocket error:', error);
                setIsBusy(false);
                isStreamingRef.current = false;
                streamingCallbackRef.current = null;
            };
            
            webSocketRef.current.onclose = () => {
                console.log('WebSocket connection closed');
                setIsBusy(false);
                isStreamingRef.current = false;
                streamingCallbackRef.current = null;
                webSocketRef.current = null;
            };
        } catch (error) {
            console.error('Error setting up WebSocket:', error);
            setIsBusy(false);
            isStreamingRef.current = false;
            streamingCallbackRef.current = null;
            return;
        }
    }, [engine]);

    const stopStreaming = useCallback(async () => {
        console.log('Stopping streaming and cleaning up resources');
        
        // Set streaming flag to false first to prevent processing new messages
        isStreamingRef.current = false;
        
        // Clear the streaming callback to prevent further updates
        const currentCallback = streamingCallbackRef.current;
        streamingCallbackRef.current = null;
        
        // Close WebSocket connection if it exists
        if (webSocketRef.current) {
            const ws = webSocketRef.current;
            webSocketRef.current = null; // Clear the reference first to prevent race conditions
            
            if (ws.readyState === WebSocket.OPEN) {
                try {
                    console.log('Closing WebSocket connection');
                    ws.close();
                } catch (error) {
                    console.error('Error closing WebSocket:', error);
                }
            }
        }
        
        // Stop any ongoing media recording
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            try {
                console.log('Stopping media recorder');
                mediaRecorderRef.current.stop();
            } catch (error) {
                console.error('Error stopping media recorder:', error);
            }
        }
        
        // Stop all media tracks
        if (streamRef.current) {
            console.log('Stopping all media tracks');
            streamRef.current.getTracks().forEach(track => {
                track.stop();
                track.enabled = false;
            });
            streamRef.current = null;
        }
        
        // Reset audio chunks array
        audioChunksRef.current = [];
        
        // Stop server-side transcription
        try {
            console.log('Sending stop request to server');
            await axios.post('http://localhost:8000/transcriber/stop');
            console.log('Server transcription stopped');
        } catch (error) {
            console.error('Error stopping server transcription:', error);
        }
        
        // Finalize the transcript (mark as not busy)
        setTranscript(prev => prev ? {
            ...prev,
            isBusy: false
        } : undefined);
        
        // Reset recording state
        setIsRecording(false);
        setIsBusy(false);
        
        console.log('Streaming and transcription fully stopped');
    }, []);
    
    // Recording functions - FIXED VERSION
    const startRecording = useCallback(async () => {
        try {
            setIsBusy(true);
            
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            
            // Store references for cleanup
            mediaRecorderRef.current = mediaRecorder;
            streamRef.current = stream;
            
            // Reset audio chunks
            audioChunksRef.current = [];
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                    console.log(`Recorded audio chunk: ${event.data.size} bytes`);
                }
            };
            
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                const audioFile = new File([audioBlob], `recording-${Date.now()}.webm`, { type: 'audio/webm' });
                
                // Create a preview URL
                const url = URL.createObjectURL(audioBlob);
                setRecordedMedia({ url, type: 'audio' });
                
                console.log('Recording stopped, transcribing file');
                await transcribeFile(audioFile);
            };
            
            // Start recording - collect data every second
            mediaRecorder.start(1000);
            setIsRecording(true);
            console.log('Audio recording started');
            
            // If live mode is enabled, start streaming
            if (isLiveMode) {
                console.log('Live mode enabled, starting streaming transcription');
                startStreaming((text, segments) => {
                    // Only update if we're still recording and streaming is active
                    if (isRecording && isStreamingRef.current) {
                        // Don't accumulate text in live mode - just show current transcription
                        setTranscript(prev => ({
                            isBusy: true,
                            text: text, // Show current text only
                            chunks: segments ? segments.map(seg => ({
                                text: seg.text,
                                start: seg.start,
                                end: seg.end,
                                timestamp: [seg.start || 0, seg.end || null]
                            })) : [],
                        }));
                    }
                });
            }
        } catch (error) {
            console.error('Error starting recording:', error);
        } finally {
            setIsBusy(false);
        }
    }, [isLiveMode, startStreaming, transcribeFile, isRecording]);
    
    const stopRecording = useCallback(async () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            try {
                console.log('Stopping media recorder and cleaning up');
                const recorder = mediaRecorderRef.current;
                
                // Stop streaming first if live mode was enabled
                if (isLiveMode && isStreamingRef.current) {
                    console.log('Stopping live streaming');
                    await stopStreaming();
                }
                
                // Stop the media recorder
                return new Promise<void>((resolve) => {
                    const onStop = () => {
                        recorder.removeEventListener('stop', onStop);
                        console.log('Media recorder stopped');
                        
                        // Stop all tracks in the stream
                        if (streamRef.current) {
                            streamRef.current.getTracks().forEach(track => {
                                track.stop();
                                track.enabled = false;
                            });
                            streamRef.current = null;
                        }
                        
                        setIsRecording(false);
                        console.log('Recording stopped');
                        resolve();
                    };
                    
                    recorder.addEventListener('stop', onStop);
                    recorder.stop();
                });
            } catch (error) {
                console.error('Error stopping recording:', error);
                setIsRecording(false);
                return Promise.resolve();
            }
        } else {
            console.log('No active recording to stop');
            return Promise.resolve();
        }
    }, [isLiveMode, stopStreaming]);
    
    // Screen recording functions
    const startScreenRecording = useCallback(async () => {
        try {
            setIsBusy(true);
            console.log('Starting screen recording, live mode:', isLiveMode);
            
            // Display a message to the user about enabling system audio
            console.info('IMPORTANT: When sharing your screen, please select "Share system audio" option if available');
            
            // Request screen sharing with audio (this will capture system OUTPUT audio on supported browsers)
            // First request system audio specifically with the displayMedia
            const screenStream = await navigator.mediaDevices.getDisplayMedia({
                video: { 
                    frameRate: { ideal: 30 },
                    width: { ideal: 1920 },
                    height: { ideal: 1080 } 
                },
                audio: {
                    // Explicitly request system audio capture
                    // This ensures we're trying to get system audio, not just any audio
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false
                } 
            });

            // Check if we actually got any system audio tracks
            let combinedStream = screenStream;
            const hasScreenAudio = screenStream.getAudioTracks().length > 0;
            
            if (hasScreenAudio) {
                console.log('Successfully captured system audio', screenStream.getAudioTracks());
            } else {
                console.warn('No system audio tracks captured. The browser might not support system audio capture.');
            }
            
            try {
                // Get microphone audio separately
                console.log('Adding microphone audio to capture voice input');
                const micAudioStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: false,
                        noiseSuppression: false,
                        autoGainControl: false
                    }
                });
                
                // Create a new stream with all tracks
                const allTracks = [
                    ...screenStream.getVideoTracks()
                ];
                
                // Add microphone audio
                if (micAudioStream.getAudioTracks().length > 0) {
                    allTracks.push(...micAudioStream.getAudioTracks());
                    console.log('Added microphone audio track');
                }
                
                // Add screen audio track if it exists (system audio)
                if (hasScreenAudio) {
                    allTracks.push(...screenStream.getAudioTracks());
                    console.log('Combined both system audio and microphone audio');
                } else {
                    console.log('No system audio detected, using microphone audio only');
                    
                    // Show a warning to the user that system audio isn't being captured
                    alert('System audio capture is not available. Only microphone audio will be recorded. Please make sure you have enabled "Share system audio" in the browser sharing dialog.');
                }
                
                combinedStream = new MediaStream(allTracks);
            } catch (audioErr) {
                console.warn('Could not get microphone audio:', audioErr);
                // Continue with just screen audio if available
                if (hasScreenAudio) {
                    console.log('Continuing with only system output audio');
                } else {
                    console.warn('No audio tracks available for recording');
                    alert('No audio tracks available for recording. Please ensure you have allowed audio access.');
                }
            }
            
            // Create a MediaRecorder for the screen + audio
            const mediaRecorder = new MediaRecorder(combinedStream, {
                mimeType: 'video/webm'
            });
            
            // Reset the chunks array
            audioChunksRef.current = [];
            
            // Store references for cleanup
            mediaRecorderRef.current = mediaRecorder;
            streamRef.current = combinedStream;
            
            // Set up event handlers
            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    audioChunksRef.current.push(e.data);
                    console.log(`Recorded chunk: ${e.data.size} bytes`);
                }
            };
            
            mediaRecorder.onstop = async () => {
                console.log('Screen recording stopped');
                // Create a blob from recorded chunks
                const blob = new Blob(audioChunksRef.current, { type: 'video/webm' });
                console.log(`Created blob: ${blob.size} bytes`);
                
                const file = new File([blob], `screen-recording-${Date.now()}.webm`, { type: 'video/webm' });
                
                // Create object URL for preview
                const url = URL.createObjectURL(blob);
                setRecordedMedia({ url, type: 'video' });
                
                // Transcribe the recording
                await transcribeFile(file);
            };
            
            // Start recording
            mediaRecorder.start(1000); // Collect data every second
            setIsRecording(true);
            console.log('MediaRecorder started');
            
            // For live mode, set up streaming
            if (isLiveMode) {
                console.log('Live mode enabled, starting streaming');
                startStreaming((text) => {
                    if (isStreamingRef.current) {
                        setTranscript(prev => ({
                            isBusy: true,
                            text: text, // Show current text, don't accumulate
                            chunks: [],
                        }));
                    }
                });
            }
        } catch (error) {
            console.error('Error starting screen recording:', error);
        } finally {
            setIsBusy(false);
        }
    }, [isLiveMode, startStreaming, transcribeFile]);

    const stopScreenRecording = useCallback(() => {
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stop();
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }
        }
        
        // Stop streaming if live mode was enabled
        if (isLiveMode) {
            stopStreaming();
        }
        
        setIsRecording(false);
        console.log('Recording stopped');
    }, [isLiveMode, stopStreaming]);

    const transcriber = useMemo((): Transcriber => {
        return {
            isBusy,
            isModelLoading,
            progressItems,
            model,
            setModel,
            language,
            setLanguage,
            engine,
            setEngine,
            isRecording,
            isLiveMode,
            setIsLiveMode,
            recordedMedia,
            transcribeFile,
            startStreaming,
            stopStreaming,
            startRecording,
            stopRecording,
            startScreenRecording,
            stopScreenRecording,
            onInputChange,
            start: postRequest,
            output: transcript,
            multilingual,
            setMultilingual,
            quantized,
            setQuantized,
            subtask,
            setSubtask,
        };
    }, [
        isBusy,
        isModelLoading,
        progressItems,
        model,
        language,
        engine,
        isRecording,
        isLiveMode,
        recordedMedia,
        transcribeFile,
        startStreaming,
        stopStreaming,
        startRecording,
        stopRecording,
        startScreenRecording,
        stopScreenRecording,
        onInputChange,
        postRequest,
        transcript,
        multilingual,
        setMultilingual,
        quantized,
        setQuantized,
        subtask,
        setSubtask,
    ]);

    return transcriber;
}