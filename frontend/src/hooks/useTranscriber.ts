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

export interface TranscriberData {
    isBusy: boolean;
    text: string;
    chunks: { text: string; start?: number; end?: number; timestamp?: [number, number | null] }[];
}

// Define the return type for transcription results
interface TranscriptionResult {
    transcript?: string;
    summary?: string;
    error?: string;
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
    
    // WebSocket for streaming transcription
    const webSocketRef = useRef<WebSocket | null>(null);
    // Audio context and media recorder for live whisper transcription
    const audioContextRef = useRef<AudioContext | null>(null);
    const micStreamRef = useRef<MediaStream | null>(null);
    const streamingCallbackRef = useRef<((text: string) => void) | null>(null);
    
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
        let transcriptText: string = ''; 
        try {
            console.log(`Transcribing file: ${file.name} with engine: ${engine}`);
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('engine', engine);
            
            const endpoint = file.type.startsWith('video/') 
                ? 'http://localhost:8000/transcribe/screen'
                : 'http://localhost:8000/transcribe/file';
            
            console.log(`Using endpoint: ${endpoint} for file: ${file.name}`);
            
            const response = await axios.post(endpoint, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            
            console.log('Transcription response:', response.data);
            
            if (response.data) {
                transcriptText = response.data.transcript || '';
                console.log('Transcript received, length:', transcriptText.length);
                
                setTranscript({
                    isBusy: false, // Set isBusy false here upon successful data handling
                    text: transcriptText,
                    chunks: [],
                });

                if (response.data.error) {
                    console.error('Transcription service returned an error:', response.data.error);
                    return { error: response.data.error }; // isBusy was set to false by setTranscript
                }
                return { transcript: transcriptText }; // isBusy was set to false by setTranscript
            } else {
                console.error('No data received from transcription service.');
                // isBusy will be set to false in finally block for this path
                return { error: 'No data received from transcription service' };
            }
        } catch (error) {
            console.error('Error transcribing file:', error);
            if (axios.isAxiosError(error) && error.response && error.response.data && error.response.data.error) {
                return { error: error.response.data.error };
            }
            return { error: 'Failed to transcribe file' };
        } finally {
            // This ensures isBusy is set to false if an error occurred before setTranscript(isBusy:false)
            // or if response.data was null.
            setIsBusy(false);
        }
    }, [engine, setTranscript, setIsBusy]); // Added setTranscript and setIsBusy to dependencies for correctness
    
    // Start streaming audio to the WebSocket for real-time transcription
    const startStreaming = useCallback(async (updateCallback: (text: string, segments?: Array<{text: string, start?: number, end?: number}>) => void, options?: {
        captureBothIO?: boolean;
        inputDevice?: string | number | null;
        outputDevice?: string | number | null;
    }) => {
        console.log('Starting streaming transcription with engine:', engine);
        
        // Prevent starting multiple streams
        if (webSocketRef.current) {
            console.warn('WebSocket already active');
            return;
        }
        
        setIsBusy(true);
        
        try {
            // First start the transcription service on the backend
            const response = await fetch(`http://localhost:8000/transcriber/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_name: model,
                    language: language || null,
                    engine: engine,
                }),
            });
            
            const data = await response.json();
            console.log('Transcription service response:', data);
            
            if (data.status !== 'started' && data.status !== 'ready') {
                throw new Error(`Failed to start transcription service: ${data.status}`);
            }
            
            // Initialize the WebSocket connection
            const ws = new WebSocket(`ws://localhost:8000/transcriber/ws`);
            webSocketRef.current = ws;
            
            // WebSocket event handlers
            ws.onopen = async () => {
                console.log('WebSocket connection established');
                
                // Send configuration to the WebSocket
                ws.send(JSON.stringify({
                    engine: engine,
                    model_name: model,
                    language: language || null
                }));
                
                // If using OpenAI Whisper, set up audio recording and streaming
                if (engine === 'whisper') {
                    try {
                        console.log('Setting up audio for Whisper streaming');
                        
                        // Get microphone access
                        const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        micStreamRef.current = micStream;
                        
                        // Create audio context
                        const audioContext = new AudioContext();
                        audioContextRef.current = audioContext;
                        
                        // Create media recorder for sending audio chunks
                        const recorder = new MediaRecorder(micStream, {
                            mimeType: 'audio/webm'
                        });
                        mediaRecorderRef.current = recorder;
                        
                        // Start recording and sending chunks
                        recorder.ondataavailable = (event) => {
                            if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
                                // Convert to ArrayBuffer and send via WebSocket
                                event.data.arrayBuffer().then(buffer => {
                                    ws.send(buffer);
                                });
                            }
                        };
                        
                        // Start the recorder with small time slices for near real-time processing
                        recorder.start(1000); // Process in 1-second chunks
                        console.log('Started Whisper audio streaming');
                    } catch (err) {
                        console.error('Error setting up audio streaming:', err);
                        setError('Failed to access microphone for streaming');
                    }
                }
            };
            
            // Handle messages from the WebSocket
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log('WebSocket received:', {
                        engine,
                        hasText: !!data.text,
                        hasSegments: !!data.segments,
                        segmentsCount: data.segments?.length
                    });
                    
                    // Handle error messages
                    if (data.error) {
                        console.error('WebSocket error:', data.error);
                        setError(data.error);
                        return;
                    }
                    
                    // Process transcription text
                    if (data.text) {
                        console.log('Received transcription:', data.text);
                        console.log('Received segments:', data.segments);
                        
                        // Process segments for display
                        const processedSegments = (data.segments || []).map((segment: any) => ({
                            text: segment.text,
                            start: segment.start || segment.timestamp?.[0],
                            end: segment.end || segment.timestamp?.[1]
                        }));
                        
                        setTranscript(prev => ({
                            isBusy: true,
                            text: data.text,
                            chunks: processedSegments,
                        }));
                        
                        // Call the update callback with the text and segments
                        if (updateCallback) {
                            updateCallback(data.text, processedSegments);
                        }
                    }
                } catch (e) {
                    console.error('Error parsing WebSocket message:', e);
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                setError('WebSocket connection error');
            };
            
            ws.onclose = () => {
                console.log('WebSocket connection closed');
                cleanupAudioResources();
                setIsBusy(false);
            };
            
        } catch (error: any) {
            console.error('Error in streaming setup:', error);
            setError(`Failed to set up streaming: ${error.message}`);
            cleanupAudioResources();
            
            // Close the WebSocket connection on error
            if (webSocketRef.current) {
                webSocketRef.current.close();
                webSocketRef.current = null;
            }
            
            setIsBusy(false);
        }
    }, [model, language, engine]);
    
    // Function to clean up audio resources
    const cleanupAudioResources = useCallback(() => {
        // Stop media recorder if active
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }
        
        // Stop and close audio tracks
        if (micStreamRef.current) {
            micStreamRef.current.getTracks().forEach(track => track.stop());
            micStreamRef.current = null;
        }
        
        // Close audio context
        if (audioContextRef.current) {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }
        
        mediaRecorderRef.current = null;
    }, []);
    
    const stopStreaming = useCallback(async () => {
        console.log('Stopping streaming, audioChunks:', audioChunksRef.current.length);
        
        // Save recording if we have audio chunks and we're in live mode
        if (audioChunksRef.current.length > 0 && isLiveMode) {
            try {
                // Create a blob from recorded chunks based on the recorder type (audio/video)
                const mimeType = mediaRecorderRef.current?.mimeType || 'audio/webm';
                const isVideo = mimeType.includes('video');
                const blob = new Blob(audioChunksRef.current, { type: mimeType });
                
                console.log(`Created ${isVideo ? 'video' : 'audio'} blob: ${blob.size} bytes`);
                
                // Create a file from the blob
                const fileName = `${isVideo ? 'screen' : 'audio'}-recording-${Date.now()}.webm`;
                const file = new File([blob], fileName, { type: mimeType });
                
                // Create object URL for preview
                const url = URL.createObjectURL(blob);
                setRecordedMedia({ url, type: isVideo ? 'video' : 'audio' });
                
                console.log(`Live recording saved as ${fileName}`);
                
                // If we have transcript content, don't overwrite it with transcription
                if (!transcript?.text) {
                    // Transcribe the recording in background
                    transcribeFile(file).catch(err => {
                        console.error('Error transcribing saved recording:', err);
                    });
                }
            } catch (err) {
                console.error('Error saving live recording:', err);
            }
        }
        
        // Use cleanup function
        cleanupAudioResources();
        
        // Close WebSocket connection
        if (webSocketRef.current) {
            webSocketRef.current.close();
            webSocketRef.current = null;
        }
        
        // Reset audio chunks array
        audioChunksRef.current = [];
        
        // Stop server-side transcription
        try {
            await axios.post('http://localhost:8000/transcriber/stop');
            console.log('Transcription stopped');
            streamingCallbackRef.current = null;
        } catch (error) {
            console.error('Error stopping transcription:', error);
        } finally {
            setIsBusy(false);
        }
    }, [isLiveMode, transcript, transcribeFile]);
    
    // Recording functions
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
                startStreaming((text) => {
                    setTranscript(prev => ({
                        isBusy: true,
                        text: prev?.text ? `${prev.text}\n${text}` : text,
                        chunks: [],
                    }));
                });
            }
        } catch (error) {
            console.error('Error starting recording:', error);
        } finally {
            setIsBusy(false);
        }
    }, [isLiveMode, startStreaming, transcribeFile]);
    
    const stopRecording = useCallback(() => {
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stop();
            
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }
            
            // Stop streaming if live mode was enabled
            if (isLiveMode) {
                stopStreaming();
            }
            
            setIsRecording(false);
            console.log('Recording stopped');
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
            const screenStream = await navigator.mediaDevices.getDisplayMedia({
                video: { 
                    frameRate: { ideal: 30 },
                    width: { ideal: 1920 },
                    height: { ideal: 1080 } 
                },
                audio: true // Try to capture system audio output
            });

            // Always try to get microphone input audio as well to capture both input and output
            let combinedStream = screenStream;
            const hasScreenAudio = screenStream.getAudioTracks().length > 0;
            
            try {
                // Get microphone audio to capture system input
                console.log('Adding microphone audio to capture all audio sources');
                const micAudioStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: false, // Disable echo cancellation for better quality
                        noiseSuppression: false, // Disable noise suppression for better quality
                        autoGainControl: false // Disable auto gain for better quality
                    }
                });
                
                // Combine all audio and video tracks
                const allTracks = [
                    ...screenStream.getVideoTracks(),
                    ...micAudioStream.getAudioTracks()
                ];
                
                // Add screen audio track if it exists
                if (hasScreenAudio) {
                    allTracks.push(...screenStream.getAudioTracks());
                    console.log('Combined both system output and microphone input audio');
                } else {
                    console.log('No system audio detected, using microphone audio only');
                }
                
                combinedStream = new MediaStream(allTracks);
            } catch (audioErr) {
                console.warn('Could not get microphone audio:', audioErr);
                // Continue with just screen audio if available
                if (hasScreenAudio) {
                    console.log('Continuing with only system output audio');
                } else {
                    console.warn('No audio tracks available for recording');
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
                    setTranscript(prev => ({
                        isBusy: true,
                        text: prev?.text ? `${prev.text}\n${text}` : text,
                        chunks: [],
                    }));
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
        }
        
        // Stop streaming if live mode was enabled
        if (isLiveMode) {
            stopStreaming();
        }
        
        setIsRecording(false);
        console.log('Recording stopped');
    }
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
