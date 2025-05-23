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

// Enhanced audio stream interface
interface AudioStreamInfo {
    hasSystemAudio: boolean;
    hasMicAudio: boolean;
    streamType: 'mic-only' | 'system-only' | 'combined';
    quality: 'low' | 'medium' | 'high';
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
    
    // Enhanced audio info
    audioStreamInfo: AudioStreamInfo;
    
    // File transcription
    transcribeFile: (file: File) => Promise<TranscriptionResult>;
    
    // Enhanced live streaming functionality
    startStreaming: (updateCallback: (text: string, segments?: Array<{text: string, start?: number, end?: number}>) => void, options?: {
        captureBothIO?: boolean;
        inputDevice?: string | number | null;
        outputDevice?: string | number | null;
        preferSystemAudio?: boolean;
    }) => void;
    stopStreaming: () => void;
    
    // Enhanced recording functionality
    startRecording: (options?: { captureSystemAudio?: boolean }) => void;
    stopRecording: () => void;
    
    // Enhanced screen recording
    startScreenRecording: (options?: { 
        preferSystemAudio?: boolean;
        combineWithMic?: boolean;
        quality?: 'low' | 'medium' | 'high';
    }) => void;
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
    
    // Enhanced audio stream tracking
    const [audioStreamInfo, setAudioStreamInfo] = useState<AudioStreamInfo>({
        hasSystemAudio: false,
        hasMicAudio: false,
        streamType: 'combined',
        quality: 'medium'
    });
    
    // Engine selection - use faster_whisper for both modes for consistency
    const [engine, setEngineState] = useState<string>("faster_whisper");
    
    // Update engine based on live mode
    useEffect(() => {
        const newEngine = "faster_whisper"; // Use faster_whisper for both modes
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
    
    // Enhanced WebSocket and audio management
    const webSocketRef = useRef<WebSocket | null>(null);
    const isStreamingRef = useRef<boolean>(false);
    const audioContextRef = useRef<AudioContext | null>(null);
    const micStreamRef = useRef<MediaStream | null>(null);
    const systemStreamRef = useRef<MediaStream | null>(null);
    const combinedStreamRef = useRef<MediaStream | null>(null);
    const streamingCallbackRef = useRef<((text: string, segments?: Array<{text: string, start?: number, end?: number}>) => void) | null>(null);
    
    // Media recorder for enhanced audio/screen recording
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const [error, setError] = useState<string>('');

    // Enhanced audio utility functions
    const createAudioContext = useCallback(() => {
        if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        }
        return audioContextRef.current;
    }, []);

    const detectAudioCapabilities = useCallback(async (): Promise<{canCaptureSystemAudio: boolean, canCaptureMic: boolean}> => {
        let canCaptureSystemAudio = false;
        let canCaptureMic = false;

        // Test microphone access
        try {
            const testMicStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            canCaptureMic = true;
            testMicStream.getTracks().forEach(track => track.stop());
        } catch (error) {
            console.warn('Microphone access not available:', error);
        }

        // Test system audio capture by checking if getDisplayMedia is a function
        // and if the browser supports audio capture (this is the most reliable way)
        try {
            if (navigator.mediaDevices && 
                typeof navigator.mediaDevices.getDisplayMedia === 'function') {
                // Check if the browser supports audio in getDisplayMedia
                // by creating a test stream (this won't actually request permissions)
                const testStream = await navigator.mediaDevices.getDisplayMedia({ 
                    video: true,
                    audio: true 
                }).catch(() => null);
                
                if (testStream) {
                    // Check if we actually got an audio track
                    const hasAudio = testStream.getAudioTracks().length > 0;
                    testStream.getTracks().forEach(track => track.stop());
                    canCaptureSystemAudio = hasAudio;
                }
            }
        } catch (error) {
            console.warn('System audio capture not available:', error);
            canCaptureSystemAudio = false;
        }

        return { canCaptureSystemAudio, canCaptureMic };
    }, []);

    const combineAudioStreams = useCallback(async (
        micStream: MediaStream | null, 
        systemStream: MediaStream | null,
        options?: { micGain?: number; systemGain?: number }
    ): Promise<MediaStream> => {
        console.log('Enhanced audio stream combination:', { 
            micTracks: micStream?.getAudioTracks().length || 0, 
            systemTracks: systemStream?.getAudioTracks().length || 0,
            options 
        });

        if (!micStream && !systemStream) {
            throw new Error('No audio streams available to combine');
        }

        // Update audio stream info
        const newStreamInfo: AudioStreamInfo = {
            hasSystemAudio: !!(systemStream?.getAudioTracks().length),
            hasMicAudio: !!(micStream?.getAudioTracks().length),
            streamType: 'combined',
            quality: 'high'
        };

        // If only one stream is available, return it directly
        if (!micStream && systemStream) {
            console.log('Using system audio only');
            newStreamInfo.streamType = 'system-only';
            setAudioStreamInfo(newStreamInfo);
            return systemStream;
        }
        
        if (micStream && !systemStream) {
            console.log('Using microphone audio only');
            newStreamInfo.streamType = 'mic-only';
            setAudioStreamInfo(newStreamInfo);
            return micStream;
        }

        // Both streams are available - combine them with enhanced processing
        try {
            const audioContext = createAudioContext();
            
            // Resume audio context if suspended
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }
            
            // Create audio destination for combined output
            const destination = audioContext.createMediaStreamDestination();
            
            // Connect microphone audio with gain control
            if (micStream && micStream.getAudioTracks().length > 0) {
                const micSource = audioContext.createMediaStreamSource(micStream);
                const micGain = audioContext.createGain();
                micGain.gain.value = options?.micGain || 1.0; // Configurable mic gain
                
                // Add mild compression for mic to even out levels
                const micCompressor = audioContext.createDynamicsCompressor();
                micCompressor.threshold.value = -24;
                micCompressor.knee.value = 30;
                micCompressor.ratio.value = 12;
                micCompressor.attack.value = 0.003;
                micCompressor.release.value = 0.25;
                
                micSource.connect(micCompressor);
                micCompressor.connect(micGain);
                micGain.connect(destination);
                console.log('Connected microphone audio with compression and gain control');
            }
            
            // Connect system audio with gain control
            if (systemStream && systemStream.getAudioTracks().length > 0) {
                const systemSource = audioContext.createMediaStreamSource(systemStream);
                const systemGain = audioContext.createGain();
                systemGain.gain.value = options?.systemGain || 1.2; // Boost system audio slightly
                
                systemSource.connect(systemGain);
                systemGain.connect(destination);
                console.log('Connected system audio with gain boost');
            }
            
            newStreamInfo.streamType = 'combined';
            newStreamInfo.quality = 'high';
            setAudioStreamInfo(newStreamInfo);
            
            return destination.stream;
            
        } catch (error) {
            console.error('Error combining audio streams with enhanced processing:', error);
            // Fallback to microphone if combination fails
            console.log('Falling back to microphone audio only');
            newStreamInfo.streamType = 'mic-only';
            setAudioStreamInfo(newStreamInfo);
            return micStream!;
        }
    }, [createAudioContext]);

    // Clean up enhanced audio resources
    const cleanupAudioResources = useCallback(() => {
        console.log('Cleaning up enhanced audio resources');
        
        // Stop all stream references
        [micStreamRef, systemStreamRef, combinedStreamRef, streamRef].forEach(ref => {
            if (ref.current) {
                ref.current.getTracks().forEach(track => {
                    track.stop();
                    track.enabled = false;
                });
                ref.current = null;
            }
        });
        
        // Close audio context
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            audioContextRef.current.close().catch(console.error);
            audioContextRef.current = null;
        }
        
        // Reset audio stream info
        setAudioStreamInfo({
            hasSystemAudio: false,
            hasMicAudio: false,
            streamType: 'combined',
            quality: 'medium'
        });
    }, []);

    // Clean up audio resources when component unmounts
    useEffect(() => {
        return () => {
            if (webSocketRef.current) {
                webSocketRef.current.close();
            }
            cleanupAudioResources();
        };
    }, [cleanupAudioResources]);

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
                console.log('Starting enhanced audio transcription process');
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
                    console.log('Processing stereo audio to mono with enhanced quality');
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

                console.log('Sending audio to worker with enhanced config:', {
                    model,
                    multilingual,
                    quantized,
                    subtask: multilingual ? subtask : null,
                    language: multilingual && language !== "auto" ? language : null,
                    audioStreamInfo
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
        [webWorker, model, multilingual, quantized, subtask, language, audioStreamInfo],
    );

    // Enhanced file transcription function
    const transcribeFile = useCallback(async (file: File): Promise<TranscriptionResult> => {
      setIsBusy(true);
      setTranscript({ text: '', chunks: [], isBusy: true });
      try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('engine', engine);
        
        const endpoint = 'http://localhost:8000/transcribe'; 
        
        console.log(`Enhanced transcribing file with ${engine} engine: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
        
        const response = await axios.post(endpoint, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 300000, // 5 minute timeout for large files
        });
        
        if (response.data && response.data.transcript !== undefined) {
          const transcriptText = response.data.transcript || '';
          const responseChunks = response.data.chunks || [];
          
          console.log(`Transcription completed: ${transcriptText.length} chars, ${responseChunks.length} segments`);
          
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

    // Enhanced live streaming functionality
    const startStreaming = useCallback(async (
        updateCallback: (text: string, segments?: Array<{text: string, start?: number, end?: number}>) => void, 
        options?: {
            captureBothIO?: boolean;
            inputDevice?: string | number | null;
            outputDevice?: string | number | null;
            preferSystemAudio?: boolean;
        }
    ) => {
        console.log('Starting enhanced streaming transcription with engine:', engine, 'options:', options);
        
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
            // Create a new WebSocket connection with enhanced parameters
            const wsUrl = `ws://localhost:8000/transcriber/ws?engine=faster-whisper&quality=high`;
            webSocketRef.current = new WebSocket(wsUrl);
            
            webSocketRef.current.onopen = () => {
                console.log('Enhanced WebSocket connection established');
                setIsBusy(false);
                
                // Send enhanced configuration
                if (webSocketRef.current) {
                    webSocketRef.current.send(JSON.stringify({
                        engine: 'faster_whisper',
                        model_name: 'base',
                        language: 'en',
                        enhance_audio: true,
                        audio_stream_info: audioStreamInfo
                    }));
                }
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
                        setTranscript(prev => ({
                            isBusy: true,
                            text: data.text,
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
                    console.error('Error processing enhanced WebSocket message:', error);
                }
            };
            
            webSocketRef.current.onerror = (error) => {
                console.error('Enhanced WebSocket error:', error);
                setIsBusy(false);
                isStreamingRef.current = false;
                streamingCallbackRef.current = null;
            };
            
            webSocketRef.current.onclose = () => {
                console.log('Enhanced WebSocket connection closed');
                setIsBusy(false);
                isStreamingRef.current = false;
                streamingCallbackRef.current = null;
                webSocketRef.current = null;
            };
        } catch (error) {
            console.error('Error setting up enhanced WebSocket:', error);
            setIsBusy(false);
            isStreamingRef.current = false;
            streamingCallbackRef.current = null;
            return;
        }
    }, [engine, audioStreamInfo]);

    const stopStreaming = useCallback(async () => {
        console.log('Stopping enhanced streaming and cleaning up resources');
        
        // Set streaming flag to false first to prevent processing new messages
        isStreamingRef.current = false;
        
        // Clear the streaming callback to prevent further updates
        streamingCallbackRef.current = null;
        
        // Close WebSocket connection if it exists
        if (webSocketRef.current) {
            const ws = webSocketRef.current;
            webSocketRef.current = null;
            
            if (ws.readyState === WebSocket.OPEN) {
                try {
                    console.log('Closing enhanced WebSocket connection');
                    ws.close();
                } catch (error) {
                    console.error('Error closing enhanced WebSocket:', error);
                }
            }
        }
        
        // Stop any ongoing media recording
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            try {
                console.log('Stopping enhanced media recorder');
                mediaRecorderRef.current.stop();
            } catch (error) {
                console.error('Error stopping enhanced media recorder:', error);
            }
        }
        
        // Clean up all audio resources
        cleanupAudioResources();
        
        // Reset audio chunks array
        audioChunksRef.current = [];
        
        // Stop server-side transcription
        try {
            console.log('Sending enhanced stop request to server');
            await axios.post('http://localhost:8000/transcriber/stop');
            console.log('Enhanced server transcription stopped');
        } catch (error) {
            console.error('Error stopping enhanced server transcription:', error);
        }
        
        // Finalize the transcript
        setTranscript(prev => prev ? {
            ...prev,
            isBusy: false
        } : undefined);
        
        // Reset recording state
        setIsRecording(false);
        setIsBusy(false);
        
        console.log('Enhanced streaming and transcription fully stopped');
    }, [cleanupAudioResources]);
    
    // Enhanced recording functions
    const startRecording = useCallback(async (options?: { captureSystemAudio?: boolean }) => {
        try {
            setIsBusy(true);
            console.log('Starting enhanced recording with options:', options);
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    sampleRate: 44100,
                    channelCount: 1
                }
            });
            
            const mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus',
                audioBitsPerSecond: 128000
            });
            
            // Store references for cleanup
            mediaRecorderRef.current = mediaRecorder;
            streamRef.current = stream;
            micStreamRef.current = stream;
            
            // Update audio stream info
            setAudioStreamInfo(prev => ({
                ...prev,
                hasMicAudio: true,
                streamType: 'mic-only'
            }));
            
            // Reset audio chunks
            audioChunksRef.current = [];
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                    console.log(`Enhanced recorded audio chunk: ${event.data.size} bytes`);
                }
            };
            
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                const audioFile = new File([audioBlob], `enhanced-recording-${Date.now()}.webm`, { type: 'audio/webm' });
                
                // Create a preview URL
                const url = URL.createObjectURL(audioBlob);
                setRecordedMedia({ url, type: 'audio' });
                
                console.log('Enhanced recording stopped, transcribing file');
                await transcribeFile(audioFile);
            };
            
            // Start recording
            mediaRecorder.start(1000);
            setIsRecording(true);
            console.log('Enhanced audio recording started');
            
            // If live mode is enabled, start streaming
            if (isLiveMode) {
                console.log('Live mode enabled, starting enhanced streaming transcription');
                startStreaming((text, segments) => {
                    if (isRecording && isStreamingRef.current) {
                        setTranscript(prev => ({
                            isBusy: true,
                            text: text,
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
            console.error('Error starting enhanced recording:', error);
        } finally {
            setIsBusy(false);
        }
    }, [isLiveMode, startStreaming, transcribeFile, isRecording]);
    
    const stopRecording = useCallback(async () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            try {
                console.log('Stopping enhanced media recorder and cleaning up');
                const recorder = mediaRecorderRef.current;
                
                // Stop streaming first if live mode was enabled
                if (isLiveMode && isStreamingRef.current) {
                    console.log('Stopping enhanced live streaming');
                    await stopStreaming();
                }
                
                // Stop the media recorder
                return new Promise<void>((resolve) => {
                    const onStop = () => {
                        recorder.removeEventListener('stop', onStop);
                        console.log('Enhanced media recorder stopped');
                        
                        cleanupAudioResources();
                        setIsRecording(false);
                        console.log('Enhanced recording stopped');
                        resolve();
                    };
                    
                    recorder.addEventListener('stop', onStop);
                    recorder.stop();
                });
            } catch (error) {
                console.error('Error stopping enhanced recording:', error);
                setIsRecording(false);
                return Promise.resolve();
            }
        } else {
            console.log('No active enhanced recording to stop');
            return Promise.resolve();
        }
    }, [isLiveMode, stopStreaming, cleanupAudioResources]);
    
    // Enhanced screen recording functions (interface only - actual implementation in App.tsx)
    const startScreenRecording = useCallback(async (options?: { 
        preferSystemAudio?: boolean;
        combineWithMic?: boolean;
        quality?: 'low' | 'medium' | 'high';
    }) => {
        console.log('Enhanced screen recording interface ready with options:', options);
        // Implementation handled in App.tsx for better access to state management
    }, []);

    const stopScreenRecording = useCallback(() => {
        console.log('Enhanced screen recording stop interface called');
        cleanupAudioResources();
        setIsRecording(false);
    }, [cleanupAudioResources]);

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
            audioStreamInfo, // Enhanced audio info
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
        audioStreamInfo, // Enhanced audio info
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