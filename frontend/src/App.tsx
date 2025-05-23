import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useTranscriber, formatTimestamp } from "./hooks/useTranscriber";
import Transcript from "./components/Transcript";
import Progress from "./components/Progress";
// Import components directly from their definition files
import Card from "./components/Card";
import StatusCard from "./components/StatusCard";
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
// Import specific icons directly to avoid type issues
import * as LucideIcons from 'lucide-react';

// Helper function to create icon elements with styling
const Icon = (icon: keyof typeof LucideIcons, size = 20, className = '') => {
  // Use type assertion to ensure TypeScript recognizes this as a valid component
  const LucideIcon = LucideIcons[icon] as React.ComponentType<{size?: number, className?: string}>;
  return LucideIcon ? <LucideIcon size={size} className={className} /> : null;
};

const API_URL = 'http://localhost:8000';

interface ActionItem {
  task: string;
  assignee: string;
}

interface ProgressStepsProps {
  currentStep: number;
}

const ProgressSteps = ({ currentStep }: ProgressStepsProps) => {
  const steps = ['Initialize', 'Record/Upload', 'Transcribe', 'Summary', 'Actions'];
  return (
    <div className="w-full mb-8">
      <div className="w-full bg-gray-200 rounded-full h-2 mb-4">
        <div
          className="bg-[#5236ab] h-2 rounded-full transition-all duration-500"
          style={{ width: `${(currentStep / (steps.length - 1)) * 100}%` }}
        />
      </div>
      <div className="flex justify-between">
        {steps.map((step, index) => (
          <div
            key={step}
            className={`text-sm ${index <= currentStep ? 'text-[#5236ab] font-medium' : 'text-gray-400'}`}
          >
            {step}
          </div>
        ))}
      </div>
    </div>
  );
};

function App() {
  const transcriber = useTranscriber();
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  
  type ActionItemType = {
    task: string;
    assignee: string;
  };
  
  type ActionItemsState = ActionItemType[] | string | null;
  
  const [actionItems, setActionItems] = useState<ActionItemsState>(null);
  const [summary, setSummary] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isGeneratingActions, setIsGeneratingActions] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [error, setError] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isScreenRecording, setIsScreenRecording] = useState(false);
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState('');
  
  // Define a type for transcript segments with improved structure
  type TranscriptSegment = {
    text: string, 
    start?: number, 
    end?: number, 
    timestamp?: [number, number | null],
    id?: string  // Add unique ID for better deduplication
  };
  
  const [liveSegments, setLiveSegments] = useState<TranscriptSegment[]>([]);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const screenMediaRecorderRef = useRef<MediaRecorder | null>(null);
  const screenAudioChunksRef = useRef<Blob[]>([]);
  const [uploadedAudioFile, setUploadedAudioFile] = useState<File | null>(null);
  
  // Enhanced audio capture states
  const [systemAudioAvailable, setSystemAudioAvailable] = useState<boolean>(false);
  const [audioStreamType, setAudioStreamType] = useState<'mic-only' | 'system-only' | 'combined'>('combined');
  
  // Improved WebSocket management
  const webSocketRef = useRef<WebSocket | null>(null);
  const webSocketSessionId = useRef<string | null>(null);
  const isLiveTranscriptionActive = useRef<boolean>(false);
  const processedSegmentIds = useRef<Set<string>>(new Set());

  // Enhanced audio context for better audio processing
  const audioContextRef = useRef<AudioContext | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const systemStreamRef = useRef<MediaStream | null>(null);
  const combinedStreamRef = useRef<MediaStream | null>(null);

  // Summary Generation - now uses direct response from Ollama API
  const generateSummaryFromAPI = async (transcript: string) => {
    try {
      console.log('Starting summary generation process');
      
      // Making direct request to the API
      console.log('Sending request to API...');
      const response = await fetch(`${API_URL}/summarize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ transcript }),
      });

      if (!response.ok) {
        console.error('Failed to generate summary:', response.status, response.statusText);
        throw new Error('Failed to generate summary');
      }

      const data = await response.json();
      
      if (data.status === 'error') {
        console.error('Summary generation error:', data.detail);
        throw new Error(data.detail || 'Failed to generate summary');
      }
      
      if (data.status === 'completed') {
        console.log('Summary generation completed successfully');
        return data.summary;
      } else {
        throw new Error('Unexpected response from server');
      }
    } catch (error) {
      console.error('Summary generation failed:', error);
      throw error;
    }
  };

  const handleGenerateSummary = async () => {
    if (liveTranscript && liveTranscript.trim() !== '') {
      try {
        console.log('Starting summary generation...');
        setError('');
        setIsProcessing(true);
        
        console.log('Transcript length for summary:', liveTranscript.length);
        const summaryText = await generateSummaryFromAPI(liveTranscript);
        
        console.log('Summary received, length:', summaryText.length);
        setSummary(summaryText);
        setCurrentStep(4);
        console.log('Summary generation completed successfully');
      } catch (error) {
        const errorMessage = 'Failed to generate summary. Please try again.';
        console.error(errorMessage, error);
        setError(errorMessage);
      } finally {
        setIsProcessing(false);
        console.log('Summary generation process finished');
      }
    } else {
      console.warn('No transcript available for summary generation (liveTranscript is empty).');
      setError('No transcript available for summary generation.');
    }
  };

  // Action Items Generation - now uses direct response from Ollama API
  const generateActionItems = async (transcript: string) => {
    setIsGeneratingActions(true);
    setError('');
    
    try {
      // Make a request to the new API endpoint for extracting action items
      const response = await fetch(`${API_URL}/generate-action-items`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ transcript }),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to extract action items: ${response.status}`);
      }
      
      const result = await response.json();
      
      // Check the status and extract action items on completion
      if (result.status === 'error') {
        // Handle error response
        console.error('Error extracting action items:', result.detail);
        setError(`Error extracting action items: ${result.detail}`);
        setIsGeneratingActions(false);
        return;
      }
      
      if (result.status === 'completed') {
        const actionItemsText = result.action_items;
        console.log('Received action items:', actionItemsText);
        
        // Process the action items, which could be HTML content
        try {
          const result = parseActionItems(actionItemsText);
          if (result && result.length > 0) {
            setActionItems(result);
          } else {
            // If parsing returns empty array, use the raw text
            setActionItems(actionItemsText || '');
          }
        } catch (e) {
          // If parsing fails, just display the raw HTML
          setActionItems(actionItemsText || '');
        }
        
        // Update UI progress
        setCurrentStep(5); // Move to the final step
        console.log('Action items extracted successfully');
        
        setIsGeneratingActions(false);
        return;
      }
      
      console.error('Unexpected response status:', result.status);
      setError(`Unknown status received: ${result.status}`);
    } catch (error) {
      console.error('Error extracting action items:', error);
      setError(`Failed to extract action items: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsGeneratingActions(false);
    }
  };

  const handleGenerateActionItems = async () => {
    if (!liveTranscript || liveTranscript.trim() === '') {
      setError('No transcript available to generate action items');
      console.warn('Attempted to generate action items with no liveTranscript.');
      return;
    }
    
    setIsGeneratingActions(true);
    setError('');
    
    try {
      console.log('Generating action items from transcript length:', liveTranscript.length);
      const response = await axios.post<{ status: string; action_items?: ActionItemType[] | string; detail?: string }>(`${API_URL}/generate-action-items`, {
        transcript: liveTranscript
      });
      console.log('Action Items API Response:', JSON.stringify(response.data));
      
      if (response.data && response.data.status === 'completed' && 'action_items' in response.data) {
        const actionItemsData = response.data.action_items;
        if (Array.isArray(actionItemsData)) {
          setActionItems(actionItemsData);
        } else if (typeof actionItemsData === 'string') {
          setActionItems(actionItemsData);
        } else {
          // This case means action_items exists but is neither array nor string
          setError('Action items data received, but not in expected format (array or string).');
          setActionItems(null); // Clear previous items
        }
      } else if (response.data && response.data.status === 'error' && response.data.detail) {
        setError(response.data.detail);
        setActionItems(null); // Clear previous items
      } else {
        // This covers cases where response.data is not as expected (e.g. missing status, or unexpected structure)
        setError('Unexpected response format from server.');
        setActionItems(null); // Clear previous items
      }
    } catch (error) {
      console.error('Error generating action items:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setError(`Failed to generate action items: ${errorMessage}`);
    } finally {
      setIsGeneratingActions(false);
    }
  };

  // Add type for the line parameter in the action items parsing
  const parseActionItems = (text: string): ActionItem[] => {
    if (!text) return [];
    
    const lines = text.split('\n').filter(line => line.trim() !== '');
    return lines.map(parseActionItemLine).filter(Boolean) as ActionItem[];
  };

  const parseActionItemLine = (line: string) => {
    if (line.includes(':')) {
      const parts = line.split(':');
      const task = parts.slice(1).join(':').trim();
      const assignee = parts[0].trim();
      return { task, assignee };
    } else if (line.includes('-')) {
      const parts = line.split('-');
      if (parts.length >= 2) {
        const task = parts[1].trim();
        const assignee = parts[0].trim();
        return { task, assignee };
      }
    }
    return { task: line.trim(), assignee: 'Unassigned' };
  };

  // Helper function to apply formatting to markdown text
  const formatMarkdown = (text: string) => {
    // Simply return the text for HTML content
    return text;
  };
  
  // Audio resampling utility function
  const resampleAudio = useCallback((audioData: Float32Array, fromSampleRate: number, toSampleRate: number): Float32Array => {
    if (fromSampleRate === toSampleRate) {
      return audioData;
    }

    const ratio = fromSampleRate / toSampleRate;
    const newLength = Math.round(audioData.length / ratio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const srcIndex = i * ratio;
      const srcIndexFloor = Math.floor(srcIndex);
      const srcIndexCeil = Math.min(srcIndexFloor + 1, audioData.length - 1);
      const fraction = srcIndex - srcIndexFloor;
      
      result[i] = audioData[srcIndexFloor] * (1 - fraction) + audioData[srcIndexCeil] * fraction;
    }

    return result;
  }, []);

  // Audio context utility function
  const createAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    return audioContextRef.current;
  }, []);

  const combineAudioStreams = useCallback(async (micStream: MediaStream | null, systemStream: MediaStream | null): Promise<MediaStream> => {
    console.log('Combining audio streams:', { 
      micTracks: micStream?.getAudioTracks().length || 0, 
      systemTracks: systemStream?.getAudioTracks().length || 0 
    });

    if (!micStream && !systemStream) {
      throw new Error('No audio streams available to combine');
    }

    // If only one stream is available, return it directly
    if (!micStream && systemStream) {
      console.log('Using system audio only');
      setAudioStreamType('system-only');
      return systemStream;
    }
    
    if (micStream && !systemStream) {
      console.log('Using microphone audio only');
      setAudioStreamType('mic-only');
      return micStream;
    }

    // Both streams are available - combine them
    try {
      const audioContext = createAudioContext();
      
      // Create audio destination for combined output
      const destination = audioContext.createMediaStreamDestination();
      
      // Connect microphone audio
      if (micStream && micStream.getAudioTracks().length > 0) {
        const micSource = audioContext.createMediaStreamSource(micStream);
        const micGain = audioContext.createGain();
        micGain.gain.value = 1.0; // Full volume for mic
        micSource.connect(micGain);
        micGain.connect(destination);
        console.log('Connected microphone audio to combined stream');
      }
      
      // Connect system audio
      if (systemStream && systemStream.getAudioTracks().length > 0) {
        const systemSource = audioContext.createMediaStreamSource(systemStream);
        const systemGain = audioContext.createGain();
        systemGain.gain.value = 1.2; // Boost system audio slightly as it's often quieter
        systemSource.connect(systemGain);
        systemGain.connect(destination);
        console.log('Connected system audio to combined stream');
      }
      
      setAudioStreamType('combined');
      return destination.stream;
      
    } catch (error) {
      console.error('Error combining audio streams:', error);
      // Fallback to microphone if combination fails
      console.log('Falling back to microphone audio only');
      setAudioStreamType('mic-only');
      return micStream!;
    }
  }, [createAudioContext]);

  // Improved WebSocket connection management
  const initializeWebSocketConnection = useCallback(() => {
    if (webSocketRef.current) {
      console.log('WebSocket already exists, cleaning up first');
      cleanupWebSocket();
    }

    console.log('Initializing new WebSocket connection');
    
    const wsUrl = `ws://localhost:8000/transcriber/ws?engine=faster-whisper`;
    const ws = new WebSocket(wsUrl);
    webSocketRef.current = ws;
    isLiveTranscriptionActive.current = true;
    
    // Clear processed segments for new session
    processedSegmentIds.current.clear();
    
    ws.onopen = () => {
      console.log('WebSocket connection established');
      // Send configuration
      ws.send(JSON.stringify({
        engine: 'faster_whisper',
        model_name: 'base',
        language: 'en'
      }));
    };
    
    ws.onmessage = (event) => {
      if (!isLiveTranscriptionActive.current) {
        console.log('Ignoring WebSocket message - transcription not active');
        return;
      }

      try {
        const data = JSON.parse(event.data);
        
        // Handle heartbeat messages
        if (data.heartbeat) {
          console.log('Received heartbeat from server');
          return;
        }
        
        // Store session ID if provided
        if (data.session_id) {
          webSocketSessionId.current = data.session_id;
        }
        
        if (data.segments && Array.isArray(data.segments)) {
          console.log(`Received ${data.segments.length} new segments`);
          
          // Process new segments with improved deduplication
          const newValidSegments: TranscriptSegment[] = [];
          
          for (const segment of data.segments) {
            const segmentText = segment.text?.trim();
            const segmentStart = segment.start || 0;
            const segmentEnd = segment.end || segmentStart;
            
            // Skip empty or very short segments
            if (!segmentText || segmentText.length < 3) {
              continue;
            }
            
            // Create a robust unique ID for the segment
            const segmentId = `${segmentText}_${segmentStart.toFixed(2)}_${segmentEnd.toFixed(2)}`;
            
            // Only add if we haven't seen this segment before
            if (!processedSegmentIds.current.has(segmentId)) {
              processedSegmentIds.current.add(segmentId);
              
              newValidSegments.push({
                text: segmentText,
                start: segmentStart,
                end: segmentEnd,
                id: segmentId,
                timestamp: [segmentStart, segmentEnd]
              });
            }
          }
          
          // Update state with new segments only
          if (newValidSegments.length > 0) {
            setLiveSegments(prevSegments => {
              const updatedSegments = [...prevSegments, ...newValidSegments];
              
              // Sort by start time
              updatedSegments.sort((a, b) => (a.start || 0) - (b.start || 0));
              
              // Update live transcript text
              const fullText = updatedSegments.map(seg => seg.text).join(' ');
              setLiveTranscript(fullText);
              
              console.log(`Added ${newValidSegments.length} new segments, total: ${updatedSegments.length}`);
              return updatedSegments;
            });
          }
        }
      } catch (error) {
        console.error('Error processing WebSocket message:', error);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Connection error during live transcription');
    };
    
    ws.onclose = (event) => {
      console.log(`WebSocket connection closed: ${event.code} ${event.reason}`);
      if (isLiveTranscriptionActive.current) {
        console.log('Unexpected WebSocket closure, cleaning up');
        cleanupWebSocket();
      }
    };
  }, []);

  const cleanupWebSocket = useCallback(() => {
    console.log('Cleaning up WebSocket connection');
    
    // Mark transcription as inactive immediately
    isLiveTranscriptionActive.current = false;
    
    if (webSocketRef.current) {
      const ws = webSocketRef.current;
      
      // Remove event listeners to prevent unwanted callbacks
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      
      // Close the connection if it's still open
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify({ command: 'stop', session_id: webSocketSessionId.current }));
          ws.close(1000, 'Client initiated cleanup');
        } catch (error) {
          console.error('Error sending stop command:', error);
        }
      }
      
      webSocketRef.current = null;
    }
    
    webSocketSessionId.current = null;
    console.log('WebSocket cleanup completed');
  }, []);

  // Cleanup audio streams function
  const cleanupAudioStreams = useCallback(() => {
    console.log('Cleaning up audio streams and context');
    
    // Stop all stream tracks
    [micStreamRef, systemStreamRef, combinedStreamRef, streamRef].forEach(streamRef => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          track.stop();
          track.enabled = false;
        });
        streamRef.current = null;
      }
    });
    
    // Close audio context
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(console.error);
      audioContextRef.current = null;
    }
    
    setSystemAudioAvailable(false);
    setAudioStreamType('combined');
  }, []);

  // Effect to cleanup WebSocket on component unmount
  useEffect(() => {
    return () => {
      cleanupWebSocket();
      cleanupAudioStreams();
    };
  }, [cleanupWebSocket, cleanupAudioStreams]);

  const startAudioRecording = (withLiveTranscript: boolean): void => {
    if (isRecording) {
      console.log('Already recording audio. Ignoring request.');
      return;
    }

    console.log(`Attempting to start audio recording with live transcript: ${withLiveTranscript}`);
    setError('');
    
    // Reset all state for new recording
    audioChunksRef.current = [];
    setLiveSegments([]);
    setLiveTranscript('');
    processedSegmentIds.current.clear();
    
    setIsLiveMode(withLiveTranscript);

    // Clear previous transcript data
    transcriber.output = undefined;
    setActionItems([]); 
    setSummary('');       
    setCurrentStep(1); 

    // Initialize WebSocket for live transcription if needed
    if (withLiveTranscript) {
      console.log('Starting live transcription WebSocket');
      initializeWebSocketConnection();
    }

    navigator.mediaDevices.getUserMedia({ 
      audio: { 
        echoCancellation: true, 
        noiseSuppression: true, 
        channelCount: 1 
      }
    })
    .then(stream => {
      console.log('Microphone access granted.');
      const options = { 
        mimeType: 'audio/webm;codecs=opus', 
        audioBitsPerSecond: 128000 
      };
      
      let recorder;
      try {
        recorder = new MediaRecorder(stream, options);
      } catch (e) {
        console.error('Error creating MediaRecorder:', e);
        setError(`Failed to initialize recording: ${e instanceof Error ? e.message : 'Unknown error'}`);
        stream.getTracks().forEach(track => track.stop());
        if (withLiveTranscript) {
          cleanupWebSocket();
        }
        setIsLiveMode(false);
        setIsRecording(false);
        return;
      }
      
      mediaRecorderRef.current = recorder;
      streamRef.current = stream;

      mediaRecorderRef.current.ondataavailable = (event: BlobEvent) => {
        console.log('[App.tsx] ondataavailable called. Chunk size:', event.data?.size);
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        console.log('MediaRecorder stopped');
        setIsRecording(false);
        
        const fullAudioBlob = new Blob(audioChunksRef.current, { 
          type: mediaRecorderRef.current?.mimeType || 'audio/webm' 
        });
        
        if (fullAudioBlob.size > 0) {
          // Save audio for later transcription
          const url = URL.createObjectURL(fullAudioBlob);
          setAudioUrl(url);
          
          // Save the blob for later transcription
          setRecordedBlob(fullAudioBlob);
          
          // Set step to show Generate Transcript button
          setCurrentStep(2);
        }
        
        // Stop all tracks in the stream
        stream.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        
        // Cleanup live transcription if it was active
        if (withLiveTranscript) {
          cleanupWebSocket();
        }
      };

      try {
        if (withLiveTranscript) { 
          mediaRecorderRef.current.start(3000); // Reduced interval for live mode
          console.log('MediaRecorder started for live mode with 3000ms timeslice.');
        } else {
          mediaRecorderRef.current.start();
          console.log('MediaRecorder started for standard recording.');
        }
        setIsRecording(true);
      } catch (e) {
        console.error('Error starting MediaRecorder:', e);
        setError(`Failed to start recording: ${e instanceof Error ? e.message : 'Unknown error'}`);
        stream.getTracks().forEach(track => track.stop());
        if (withLiveTranscript) {
          cleanupWebSocket();
        }
        setIsRecording(false);
        setIsLiveMode(false);
      }
    })
    .catch(err => {
      console.error('Failed to get user media (microphone):', err);
      setError('Could not access microphone. Please check permissions.');
      setIsRecording(false);
      setIsLiveMode(false);
    });
  };

  const handleGenerateTranscript = async (uploadedFile?: File) => {
    const audioToTranscribe = uploadedFile || recordedBlob;

    if (!audioToTranscribe) {
      console.error('No audio available for transcription.');
      setError('No audio available. Please record or upload audio first.');
      return;
    }

    if (audioToTranscribe instanceof Blob) {
      setAudioUrl(URL.createObjectURL(audioToTranscribe));
    }

    setIsTranscribing(true);
    setError('');

    try {
      console.log('Starting transcription...');
      const audioFile = audioToTranscribe instanceof File ? audioToTranscribe : new File([audioToTranscribe], 'recording.webm', { type: audioToTranscribe.type });
      
      const formDataObj = new FormData(); 
      formDataObj.append('file', audioFile);
      formDataObj.append('engine', 'faster_whisper');
      
      const response = await fetch(`${API_URL}/transcribe`, {
        method: 'POST',
        body: formDataObj
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Transcription failed: ${response.status} - ${errorText}`);
      }
      
      const result = await response.json();
      console.log('Transcription API response:', result);
      
      if (result.status === 'completed') {
        // Convert chunks to segments format
        const segments = result.chunks || result.segments || [];
        const processedSegments = segments.map((chunk: any, index: number) => ({
          text: chunk.text || '',
          start: chunk.start || 0,
          end: chunk.end || 0,
          timestamp: [chunk.start || 0, chunk.end || 0],
          id: `segment_${index}_${chunk.start || 0}`
        }));
        
        setLiveSegments(processedSegments);
        setLiveTranscript(result.transcript || '');
        setCurrentStep(3);
      } else if (result.status === 'error') {
        throw new Error(result.detail || 'Transcription API returned an error.');
      } else {
        throw new Error('Unexpected response format from transcription API.');
      }
    } catch (err) {
      console.error('Error generating transcript:', err);
      setError(`Failed to generate transcript: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsTranscribing(false);
    }
  };

  // Enhanced screen recording with proper live audio streaming
  const startScreenRecording = async (withLiveTranscript: boolean) => {
    if (isScreenRecording) {
      console.log('Already screen recording.');
      return;
    }
    
    console.log(`Starting enhanced screen recording with live transcript: ${withLiveTranscript}`);
    setIsScreenRecording(true);
    setError('');
    
    // Reset state for new recording
    screenAudioChunksRef.current = [];
    setLiveSegments([]);
    setLiveTranscript('');
    processedSegmentIds.current.clear();
    cleanupAudioStreams(); // Clean up any existing streams
    
    setIsLiveMode(withLiveTranscript);

    try {
      console.log('Step 1: Requesting screen share with system audio...');
      
      // Step 1: Request screen sharing with explicit system audio request
      const displayStream = await navigator.mediaDevices.getDisplayMedia({
        video: { 
          displaySurface: 'monitor',
          frameRate: { ideal: 30 },
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        },
        audio: {
          // Request system audio explicitly
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          suppressLocalAudioPlayback: false, // Allow capturing played audio
        }
      });

      // Check what audio tracks we got from screen sharing
      const systemAudioTracks = displayStream.getAudioTracks();
      console.log(`Screen share returned ${systemAudioTracks.length} audio tracks:`, 
        systemAudioTracks.map(track => ({ 
          label: track.label,
          kind: track.kind,
          enabled: track.enabled
        }))
      );

      if (systemAudioTracks.length > 0) {
        console.log('✅ System audio captured successfully!');
        setSystemAudioAvailable(true);
        systemStreamRef.current = new MediaStream(systemAudioTracks);
      } else {
        console.warn('❌ No system audio tracks captured from screen share');
        setSystemAudioAvailable(false);
        
        // Show user instruction for better system audio capture
        const userAgent = navigator.userAgent.toLowerCase();
        let instruction = '';
        
        if (userAgent.includes('chrome')) {
          instruction = 'In Chrome: Make sure to check "Share system audio" when selecting screen/tab to share.';
        } else if (userAgent.includes('firefox')) {
          instruction = 'In Firefox: System audio capture may not be available. Try using Chrome for better system audio support.';
        } else if (userAgent.includes('safari')) {
          instruction = 'In Safari: System audio capture is not supported. Try using Chrome or Firefox.';
        } else {
          instruction = 'Make sure to enable "Share system audio" option when sharing your screen/tab.';
        }
        
        console.warn(`System audio not available. ${instruction}`);
        alert(`System audio not detected!\n\n${instruction}\n\nWe'll continue with microphone audio only.`);
      }

      console.log('Step 2: Requesting microphone access...');
      
      // Step 2: Get microphone audio separately with optimized settings
      let micStream: MediaStream | null = null;
      try {
        micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: false, // Disable to get raw audio
            noiseSuppression: false, // Disable to preserve system audio
            autoGainControl: false,  // Disable to maintain levels
            sampleRate: 44100,       // High quality sample rate
            channelCount: 1,         // Mono is sufficient for speech
          }
        });
        
        console.log(`✅ Microphone access granted: ${micStream.getAudioTracks().length} tracks`);
        micStreamRef.current = micStream;
      } catch (micError) {
        console.warn('⚠️ Could not access microphone:', micError);
        alert('Could not access microphone. Continuing with system audio only (if available).');
      }

      console.log('Step 3: Combining audio streams...');
      
      // Step 3: Combine all available audio streams
      const finalAudioStream = await combineAudioStreams(micStream, systemStreamRef.current);
      
      console.log('Step 4: Creating combined media stream...');
      
      // Step 4: Create final combined stream with video + combined audio
      const finalCombinedStream = new MediaStream();
      
      // Add video tracks from display stream
      displayStream.getVideoTracks().forEach(track => {
        finalCombinedStream.addTrack(track);
        console.log(`Added video track: ${track.label}`);
      });
      
      // Add combined audio tracks
      finalAudioStream.getAudioTracks().forEach(track => {
        finalCombinedStream.addTrack(track);
        console.log(`Added combined audio track: ${track.label || 'Combined Audio'}`);
      });
      
      combinedStreamRef.current = finalCombinedStream;

      console.log(`Final stream created with ${finalCombinedStream.getVideoTracks().length} video tracks and ${finalCombinedStream.getAudioTracks().length} audio tracks`);

      // Step 5: Initialize WebSocket for live transcription BEFORE starting recording
      if (withLiveTranscript) {
        console.log('Step 5a: Starting live transcription WebSocket for screen recording');
        initializeWebSocketConnection();
        
        // Step 5b: Set up audio processing for live transcription
        await setupLiveAudioProcessing(finalAudioStream);
      }

      console.log('Step 6: Setting up MediaRecorder...');
      
      // Step 6: Set up MediaRecorder with optimal settings
      const recorderOptions = {
        mimeType: 'video/webm;codecs=vp9,opus',
        videoBitsPerSecond: 2500000, // 2.5 Mbps for good quality
        audioBitsPerSecond: 128000,  // 128 kbps for clear audio
      };
      
      // Fallback options if the preferred format isn't supported
      const fallbackOptions = [
        { mimeType: 'video/webm;codecs=vp8,opus' },
        { mimeType: 'video/webm' },
        { mimeType: 'video/mp4' }
      ];
      
      let screenRecorder: MediaRecorder;
      try {
        screenRecorder = new MediaRecorder(finalCombinedStream, recorderOptions);
      } catch (e) {
        console.warn('Primary recorder options failed, trying fallbacks:', e);
        
        for (const options of fallbackOptions) {
          try {
            screenRecorder = new MediaRecorder(finalCombinedStream, options);
            console.log(`Using fallback recorder with: ${options.mimeType}`);
            break;
          } catch (fallbackError) {
            console.warn(`Fallback ${options.mimeType} failed:`, fallbackError);
          }
        }
        
        if (!screenRecorder!) {
          throw new Error('No supported MediaRecorder format found');
        }
      }
      
      screenMediaRecorderRef.current = screenRecorder;

      // Set up recorder event handlers
      screenRecorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          screenAudioChunksRef.current.push(event.data);
          console.log(`Recorded chunk: ${event.data.size} bytes`);
        }
      };

      screenRecorder.onstop = async () => {
        console.log('Screen recorder stopped.');
        setIsScreenRecording(false);
        
        const videoBlob = new Blob(screenAudioChunksRef.current, { 
          type: screenRecorder.mimeType || 'video/webm' 
        });
        
        if (videoBlob.size > 0) {
          console.log(`Recording completed - Size: ${(videoBlob.size / 1024 / 1024).toFixed(2)} MB`);
          
          // Create preview URL
          const url = URL.createObjectURL(videoBlob);
          setAudioUrl(url);
          
          // If this was live mode and we have segments, move to next step
          if (withLiveTranscript && liveSegments.length > 0) {
            console.log('Live recording finished with segments:', liveSegments.length);
            setCurrentStep(3);
          } else {
            // For non-live mode, transcribe the recorded file
            try {
              console.log('Starting transcription of recorded file...');
              setIsTranscribing(true);
              
              const videoFile = new File([videoBlob], `screen_recording_${Date.now()}.webm`, { 
                type: videoBlob.type 
              });
              
              const result = await transcriber.transcribeFile(videoFile);
              console.log('Transcription completed:', result);
              
              if (result.transcript) {
                setLiveTranscript(result.transcript);
                if (result.chunks) {
                  const processedSegments = result.chunks.map((chunk: any, index: number) => ({
                    text: chunk.text || '',
                    start: chunk.start || 0,
                    end: chunk.end || 0,
                    timestamp: [chunk.start || 0, chunk.end || 0] as [number, number | null],
                    id: `segment_${index}_${chunk.start || 0}`
                  }));
                  setLiveSegments(processedSegments);
                }
                setCurrentStep(3);
              }
            } catch (error) {
              console.error('Transcription error:', error);
              setError(`Failed to transcribe recording: ${error instanceof Error ? error.message : 'Unknown error'}`);
            } finally {
              setIsTranscribing(false);
            }
          }
        } else {
          console.warn('No data recorded - blob size is 0');
          setError('Recording failed: No data captured');
        }
        
        // Cleanup all streams
        cleanupAudioStreams();
        
        // Cleanup live transcription if it was active
        if (withLiveTranscript) {
          cleanupWebSocket();
        }
      };

      console.log('Step 7: Starting recording...');
      
      // Step 7: Start recording
      try {
        screenRecorder.start(1000); // Collect data every second
        console.log('✅ Screen recording started successfully!');
        
        // Show recording status with audio source info
        const audioInfo = systemAudioAvailable 
          ? (micStream ? 'System + Microphone Audio' : 'System Audio Only')
          : 'Microphone Audio Only';
        
        console.log(`🎙️ Audio capture mode: ${audioInfo}`);
        
      } catch (startError) {
        console.error('Failed to start screen recorder:', startError);
        throw startError;
      }
      
    } catch (err) {
      console.error('Error in enhanced screen recording:', err);
      let errorMessage = 'Failed to start screen recording. ';
      
      if (err instanceof Error) {
        if (err.message.includes('Permission denied')) {
          errorMessage += 'Permission denied. Please allow screen sharing and try again.';
        } else if (err.message.includes('NotAllowedError')) {
          errorMessage += 'Screen sharing was cancelled or not allowed.';
        } else if (err.message.includes('NotSupportedError')) {
          errorMessage += 'Screen recording is not supported in this browser.';
        } else {
          errorMessage += err.message;
        }
      } else {
        errorMessage += 'Unknown error occurred.';
      }
      
      setError(errorMessage);
      setIsScreenRecording(false);
      setIsLiveMode(false);
      
      // Cleanup on error
      cleanupAudioStreams();
      if (withLiveTranscript) {
        cleanupWebSocket();
      }
    }
  };

  // NEW: Send audio data to WebSocket for transcription
  const sendAudioForTranscription = useCallback((audioChunks: Float32Array[], sampleRate: number) => {
    if (!webSocketRef.current || webSocketRef.current.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not ready for audio data');
      return;
    }

    try {
      // Combine all audio chunks
      const totalLength = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
      const combinedAudio = new Float32Array(totalLength);
      
      let offset = 0;
      for (const chunk of audioChunks) {
        combinedAudio.set(chunk, offset);
        offset += chunk.length;
      }

      // Convert to 16kHz mono for Whisper (if needed)
      let processedAudio = combinedAudio;
      if (sampleRate !== 16000) {
        processedAudio = resampleAudio(combinedAudio, sampleRate, 16000);
      }

      // Convert Float32Array to ArrayBuffer
      const audioBuffer = new ArrayBuffer(processedAudio.length * 4);
      const view = new Float32Array(audioBuffer);
      view.set(processedAudio);

      // Send audio data to WebSocket
      const message = {
        type: 'audio_data',
        audio: Array.from(processedAudio), // Convert to regular array for JSON
        sample_rate: 16000,
        timestamp: Date.now()
      };

      webSocketRef.current.send(JSON.stringify(message));
      console.log(`Sent ${processedAudio.length} audio samples for live transcription`);
      
    } catch (error) {
      console.error('Error sending audio for transcription:', error);
    }
  }, []);

  // NEW: Simple audio resampling function
  const resampleAudio = useCallback((audioData: Float32Array, fromSampleRate: number, toSampleRate: number): Float32Array => {
    if (fromSampleRate === toSampleRate) {
      return audioData;
    }

    const ratio = fromSampleRate / toSampleRate;
    const newLength = Math.round(audioData.length / ratio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const srcIndex = i * ratio;
      const srcIndexFloor = Math.floor(srcIndex);
      const srcIndexCeil = Math.min(srcIndexFloor + 1, audioData.length - 1);
      const fraction = srcIndex - srcIndexFloor;
      
      result[i] = audioData[srcIndexFloor] * (1 - fraction) + audioData[srcIndexCeil] * fraction;
    }

    return result;
  }, []);

  // NEW: Setup live audio processing for real-time transcription
  const setupLiveAudioProcessing = useCallback(async (audioStream: MediaStream) => {
    if (!audioStream || audioStream.getAudioTracks().length === 0) {
      console.warn('No audio tracks available for live processing');
      return;
    }

    try {
      console.log('Setting up live audio processing for screen recording');
      
      // Create audio context for processing
      const audioContext = createAudioContext();
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }

      // Create analyser and processor nodes
      const source = audioContext.createMediaStreamSource(audioStream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      
      // Buffer to accumulate audio data
      let audioBuffer: Float32Array[] = [];
      let bufferDuration = 0;
      const targetDuration = 3; // seconds
      const sampleRate = audioContext.sampleRate;

      processor.onaudioprocess = (event) => {
        if (!isLiveTranscriptionActive.current || !isScreenRecording) {
          return;
        }

        const inputBuffer = event.inputBuffer;
        const inputData = inputBuffer.getChannelData(0);
        
        // Only process if there's actual audio (not silence)
        const audioLevel = Math.sqrt(inputData.reduce((sum, x) => sum + x * x, 0) / inputData.length);
        
        if (audioLevel > 0.001) { // Threshold to avoid processing silence
          // Copy audio data to buffer
          const chunk = new Float32Array(inputData.length);
          chunk.set(inputData);
          audioBuffer.push(chunk);
          
          bufferDuration += inputData.length / sampleRate;
          
          // When we have enough audio, send it for transcription
          if (bufferDuration >= targetDuration) {
            sendAudioForTranscription(audioBuffer, sampleRate);
            audioBuffer = [];
            bufferDuration = 0;
          }
        }
      };

      // Connect the audio processing chain
      source.connect(processor);
      processor.connect(audioContext.destination);
      
      console.log('Live audio processing setup completed');
      
      // Store references for cleanup
      audioContextRef.current = audioContext;
      
    } catch (error) {
      console.error('Error setting up live audio processing:', error);
    }
  }, [createAudioContext, sendAudioForTranscription]);

  const stopScreenRecording = useCallback(() => {
    console.log('Stopping screen recording...');
    
    // Set recording state to false immediately
    setIsScreenRecording(false);
    
    // Stop the media recorder first
    if (screenMediaRecorderRef.current) {
      if (screenMediaRecorderRef.current.state === 'recording') {
        screenMediaRecorderRef.current.stop();
        console.log('Screen recorder stopped');
      }
      screenMediaRecorderRef.current = null;
    }
    
    // Cleanup all audio streams
    cleanupAudioStreams();
    
    // Cleanup WebSocket connection
    cleanupWebSocket();
    
    setIsLiveMode(false);
    console.log('Screen recording cleanup completed');
  }, [cleanupWebSocket, cleanupAudioStreams]);

  const stopAudioRecording = (): void => {
    console.log('stopAudioRecording called. Current state:', mediaRecorderRef.current?.state);
    
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop(); // This will trigger the onstop handler
    } else {
      console.warn('Audio recorder not recording or not initialized.');
      setIsRecording(false);
      setIsLiveMode(false);
      
      // Stop any active tracks in the stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track: MediaStreamTrack) => track.stop());
        streamRef.current = null;
      }
      
      // Cleanup WebSocket if active
      cleanupWebSocket();
    }
  };

  // Effect to handle live segments when they change
  useEffect(() => {
    // Only process if we have segments and live mode is active
    if (liveSegments.length > 0 && isLiveMode) {
      console.log(`Live segments updated: ${liveSegments.length} segments`);
      
      // Update transcriber output with live segments
      if (transcriber) {
        transcriber.output = {
          isBusy: isRecording || isScreenRecording,
          text: liveTranscript,
          chunks: liveSegments.map(segment => ({
            text: segment.text,
            timestamp: [segment.start || 0, segment.end || 0] as [number, number | null]
          }))
        };
      }
    }
  }, [liveSegments, liveTranscript, isLiveMode, isRecording, isScreenRecording, transcriber]);

  // Save Functions
  const saveSummary = () => {
    const element = document.createElement("a");
    const file = new Blob([summary || ""], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = "meeting_summary.txt";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const saveActionItems = () => {
    if (!actionItems) {
      setError('No action items to save');
      return;
    }

    let content = '';
    
    if (typeof actionItems === 'string') {
      // If actionItems is a string (HTML), create a text version
      const tempDiv = document.createElement('div');
      tempDiv.innerHTML = actionItems;
      content = tempDiv.textContent || tempDiv.innerText || '';
    } else if (Array.isArray(actionItems)) {
      // If actionItems is an array of ActionItemType
      content = actionItems.map(item => 
        `Task: ${item.task}\nAssignee: ${item.assignee}\n\n`
      ).join('');
    }
    
    if (!content) {
      setError('No content to save');
      return;
    }
    
    try {
      const element = document.createElement("a");
      const file = new Blob([content], { type: 'text/plain' });
      element.href = URL.createObjectURL(file);
      element.download = "action_items.txt";
      document.body.appendChild(element);
      element.click();
      document.body.removeChild(element);
    } catch (error) {
      console.error('Error saving action items:', error);
      setError('Failed to save action items');
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h1 className="text-4xl font-extrabold text-gray-900 tracking-tight">Voxa</h1>
          <p className="mt-3 max-w-md mx-auto text-lg text-gray-500">
            Transform meetings into actionable intelligence with advanced voice recognition.
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <ProgressSteps currentStep={currentStep} />

        {/* Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <StatusCard
            icon={Icon('BarChart2')}
            title="Model Status"
            value={transcriber.isModelLoading ? "Loading..." : "Ready"}
          />
          <StatusCard
            icon={Icon('Settings')}
            title="Processing Status"
            value={(transcriber.isBusy || isProcessing || isGeneratingActions) ? "Processing" : "Idle"}
          />
          <StatusCard
            icon={Icon('Users')}
            title="Session Status"
            value={isLiveTranscriptionActive.current ? "Live Active" : (transcriber.output ? "Active" : "Waiting")}
          />
        </div>

        {/* Enhanced Audio Input Options */}
        <Card title="Input Options" icon={Icon('Mic')} fullWidth>
          <div className="space-y-4">
            {/* Live Mode Toggle */}
            <div className="flex items-center justify-end mb-2">
              <span className="mr-2 text-sm font-medium text-gray-700">Live Mode</span>
              <button 
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${isLiveMode ? 'bg-[#5236ab]' : 'bg-gray-300'}`}
                onClick={() => setIsLiveMode(!isLiveMode)}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${isLiveMode ? 'translate-x-6' : 'translate-x-1'}`} />
              </button>
            </div>

            {/* Audio Capture Status Display */}
            {(isRecording || isScreenRecording) && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
                <div className="flex items-center mb-2">
                  <div className="w-3 h-3 bg-blue-500 rounded-full mr-2 animate-pulse"></div>
                  <span className="text-sm font-medium text-blue-800">
                    Audio Capture Status
                  </span>
                </div>
                <div className="text-xs text-blue-600 space-y-1">
                  <div className="flex items-center">
                    <span className="font-medium mr-2">Mode:</span>
                    <span className="capitalize">{audioStreamType.replace('-', ' + ')}</span>
                  </div>
                  <div className="flex items-center">
                    <span className="font-medium mr-2">System Audio:</span>
                    <span className={systemAudioAvailable ? 'text-green-600' : 'text-orange-600'}>
                      {systemAudioAvailable ? '✅ Captured' : '⚠️ Not Available'}
                    </span>
                  </div>
                  <div className="flex items-center">
                    <span className="font-medium mr-2">Live Transcription:</span>
                    <span className={isLiveTranscriptionActive.current ? 'text-green-600' : 'text-red-600'}>
                      {isLiveTranscriptionActive.current ? '✅ Active' : '❌ Inactive'}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Recording options when nothing is recording */}
            {!isRecording && !isScreenRecording && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* File Upload Button */}
                <button
                  className="flex flex-col items-center justify-center p-6 border-2 border-gray-200 rounded-xl hover:border-[#5236ab] hover:bg-gray-50 transition-all duration-200"
                  onClick={() => document.getElementById('file-upload')?.click()}
                >
                  {Icon('Upload', 32, "w-8 h-8 mb-2 text-[#5236ab]")}
                  <span className="font-medium">Upload File</span>
                  <span className="text-xs text-gray-500 mt-1">Audio or Video</span>
                  <input
                    id="file-upload"
                    type="file"
                    accept="audio/*,video/*"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        setUploadedAudioFile(file);
                        setAudioUrl(URL.createObjectURL(file));
                        setRecordedBlob(null); // Clear any previous recording
                        // Reset transcript data for new file
                        setLiveSegments([]);
                        setLiveTranscript('');
                        transcriber.output = undefined;
                        setActionItems([]);
                        setSummary('');
                        setCurrentStep(2); // Show Generate Transcript button
                      }
                    }}
                  />
                </button>

                {/* Audio Recording Button */}
                <button
                  className="flex flex-col items-center justify-center p-6 border-2 border-gray-200 rounded-xl hover:border-[#5236ab] hover:bg-gray-50 transition-all duration-200"
                  onClick={() => {
                    setIsRecording(true);
                    startAudioRecording(isLiveMode);
                  }}
                >
                  {Icon('Mic', 32, "w-8 h-8 mb-2 text-[#5236ab]")}
                  <span className="font-medium">Audio Recording</span>
                  <span className="text-xs text-gray-500 mt-1">{isLiveMode ? 'With Live Transcript' : 'Standard'}</span>
                </button>

                {/* Enhanced Screen Recording Button */}
                <button
                  className="flex flex-col items-center justify-center p-6 border-2 border-gray-200 rounded-xl hover:border-[#5236ab] hover:bg-gray-50 transition-all duration-200"
                  onClick={() => {
                    setIsScreenRecording(true);
                    startScreenRecording(isLiveMode);
                  }}
                >
                  {Icon('Monitor', 32, "w-8 h-8 mb-2 text-[#5236ab]")}
                  <span className="font-medium">Screen + Audio</span>
                  <span className="text-xs text-gray-500 mt-1">
                    {isLiveMode ? 'Live Teams/Meeting Audio' : 'Record Meeting'}
                  </span>
                  <span className="text-xs text-blue-600 mt-1 font-medium">
                    Captures System + Mic Audio
                  </span>
                </button>
              </div>
            )}
            
            {/* Enhanced Stop Recording Button */}
            {(isRecording || isScreenRecording) && (
              <div className="flex flex-col items-center space-y-3">
                <button
                  onClick={() => {
                    if (isRecording) {
                      stopAudioRecording();
                    } else if (isScreenRecording) {
                      stopScreenRecording();
                    }
                  }}
                  className="bg-red-600 hover:bg-red-700 text-white font-medium py-4 px-8 rounded-lg shadow-md transition-colors duration-200 flex items-center justify-center"
                >
                  {Icon('StopCircle', 24, "w-6 h-6 mr-2")}
                  Stop {isScreenRecording ? 'Screen Recording' : 'Audio Recording'}
                </button>
                
                {/* Enhanced Recording Status */}
                <div className="text-center space-y-2">
                  <div className="flex items-center justify-center">
                    <div className="w-3 h-3 bg-red-500 rounded-full mr-2 animate-pulse"></div>
                    <span className="text-sm font-medium">
                      {isRecording ? 'Recording audio...' : 'Recording screen and audio...'}
                      {isLiveMode && ' with live transcription'}
                    </span>
                  </div>
                  
                  {/* System audio status indicator for screen recording */}
                  {isScreenRecording && (
                    <div className="text-xs text-gray-600 bg-gray-100 rounded-md px-2 py-1 inline-block">
                      {systemAudioAvailable 
                        ? '🎵 System audio + microphone being captured' 
                        : '🎤 Microphone only (enable "Share system audio" for better results)'}
                    </div>
                  )}
                  
                  {/* WebSocket connection status */}
                  {isLiveMode && (
                    <div className="flex items-center justify-center text-xs text-gray-500">
                      <div className={`w-2 h-2 rounded-full mr-1 ${
                        isLiveTranscriptionActive.current ? 'bg-green-500' : 'bg-red-500'
                      }`}></div>
                      Live transcription: {isLiveTranscriptionActive.current ? 'Connected' : 'Disconnected'}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* System Audio Tips */}
            {!isRecording && !isScreenRecording && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mt-4">
                <div className="flex items-start">
                  {Icon('Info', 20, "w-5 h-5 text-blue-500 mr-2 mt-0.5")}
                  <div>
                    <h4 className="text-sm font-medium text-blue-800 mb-1">
                      Tips for Capturing Meeting Audio
                    </h4>
                    <ul className="text-xs text-blue-600 space-y-1">
                      <li>• <strong>Teams/Zoom Meetings:</strong> Use "Screen + Audio" and select "Share system audio"</li>
                      <li>• <strong>Browser Meetings:</strong> Share the specific browser tab with audio enabled</li>
                      <li>• <strong>Best Results:</strong> Enable live mode for real-time transcription</li>
                      <li>• <strong>Audio Quality:</strong> Ensure your microphone and system volume are properly set</li>
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* Live Transcript Display - Enhanced */}
        {((isRecording || isScreenRecording) && isLiveMode) && (
          <Card title="Live Transcript with Timestamps" icon={Icon('Mic')} fullWidth>
            <div className="max-h-80 overflow-y-auto my-2 p-4 bg-gray-50 rounded-lg">
              {/* Enhanced debug info - only show in development */}
              {process.env.NODE_ENV === 'development' && (
                <div className="mb-2 p-2 bg-gray-100 text-xs rounded">
                  <div className="grid grid-cols-2 gap-1">
                    <div>Live segments: <span className="font-mono">{liveSegments.length || 0}</span></div>
                    <div>Text length: <span className="font-mono">{liveTranscript?.length || 0}</span></div>
                    <div>WebSocket: <span className="font-mono">{isLiveTranscriptionActive.current ? 'Connected' : 'Disconnected'}</span></div>
                    <div>Audio Mode: <span className="font-mono">{audioStreamType}</span></div>
                    <div>System Audio: <span className="font-mono">{systemAudioAvailable ? 'Yes' : 'No'}</span></div>
                    <div>Processed IDs: <span className="font-mono">{processedSegmentIds.current.size}</span></div>
                  </div>
                </div>
              )}
              
              {liveSegments.length > 0 ? (
                <div className="space-y-2">
                  {liveSegments.map((segment, index) => (
                    <div key={segment.id || index} className="border-b border-gray-200 pb-2 last:border-0">
                      <div className="text-sm text-gray-500 mb-1">
                        {segment.start !== undefined ? formatTimestamp(segment.start) : '00:00:00'} 
                        {segment.end !== undefined ? ` - ${formatTimestamp(segment.end)}` : ''}
                      </div>
                      <div className="text-gray-800">{segment.text}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-gray-500 italic flex items-center justify-center h-20">
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-blue-500 rounded-full mr-2 animate-pulse"></div>
                    <span>Listening for speech...</span>
                  </div>
                </div>
              )}
            </div>
          </Card>
        )}
        
        {/* Display audio player when audio is available */}
        {audioUrl && (
          <Card title="Recorded Audio" icon={Icon('Music')} fullWidth className="mb-4">
            <div className="p-4">
              <audio 
                className="w-full" 
                controls 
                src={audioUrl}
              >
                Your browser does not support the audio element.
              </audio>
              {/* Generate Transcript Button - Always show when audio is available and not in live mode */}
              {audioUrl && !isLiveMode && (
                <div className="mt-4 flex justify-center">
                  <button
                    onClick={() => handleGenerateTranscript(uploadedAudioFile || undefined)}
                    className="bg-[#5236ab] hover:bg-[#4527a0] text-white font-medium py-2 px-6 rounded-lg shadow-md transition-colors duration-200 flex items-center"
                    disabled={isTranscribing || isProcessing}
                  >
                    {Icon('FileText', 20, "w-5 h-5 mr-2")}
                    {isTranscribing ? 'Generating Transcript...' : 'Generate Transcript'}
                  </button>
                </div>
              )}
            </div>
          </Card>
        )}
        
        {/* Always display the transcript card, even if empty */}
        <Card title="Transcript" icon={Icon('FileText')} fullWidth className="mb-4">
          {(liveTranscript && liveTranscript.trim() !== '') || (liveSegments && liveSegments.length > 0) ? (
            <>
              <Transcript transcriptText={liveTranscript} segments={liveSegments} />
              <div className="mt-4 flex justify-end space-x-4">
                <button
                  onClick={handleGenerateSummary}
                  disabled={isProcessing || isTranscribing}
                  className={`flex items-center px-4 py-2 ${
                    (isProcessing || isTranscribing) ? 'bg-gray-400 cursor-not-allowed' : 'bg-[#5236ab] hover:bg-[#4527a0]'
                  } text-white rounded-md transition-colors`}
                >
                  {Icon('ChevronRight', 16, "w-4 h-4 mr-2")}
                  {isProcessing ? 'Generating Summary...' : 'Generate Summary'}
                </button>
              </div>
            </>
          ) : (
            <div className="p-4 text-center text-gray-500">
              No transcript available yet. Record audio or upload a file to start.
            </div>
          )}
        </Card>

        {/* Summary Display */}
        {summary && (
          <Card title="Meeting Summary" icon={Icon('FileText')} fullWidth className="mb-4">
            <div className="bg-gray-50 p-4 rounded-lg mb-4">
              <div className="prose max-w-none">
                {/* Format the summary with React Markdown */}
                <ReactMarkdown
                  components={{
                    // Make headings bold and properly styled
                    h1: ({ node, ...props }) => <h1 className="text-xl font-bold mt-4 mb-2" {...props} />,
                    h2: ({ node, ...props }) => <h2 className="text-lg font-bold mt-3 mb-2" {...props} />,
                    h3: ({ node, ...props }) => <h3 className="text-md font-bold mt-2 mb-1" {...props} />,
                    // Style list items
                    li: ({ node, ...props }) => <li className="ml-4 my-1" {...props} />,
                    // Style paragraphs
                    p: ({ node, ...props }) => <p className="my-2" {...props} />,
                    // Style section headers (detected by regex)
                    strong: ({ node, ...props }) => {
                      // Check if the content looks like a section header
                      const isSectionHeader = props.children && typeof props.children === 'string' &&
                        /^(\d+\.|Meeting Overview|Key Discussion Points|Action Items)/.test(props.children as string);
                      
                      return isSectionHeader ?
                        <strong className="text-lg font-bold block mt-4 mb-2 text-[#5236ab]" {...props} /> :
                        <strong className="font-bold" {...props} />;
                    },
                  }}
                >
                  {formatMarkdown(summary)}
                </ReactMarkdown>
              </div>
            </div>
            <div className="flex justify-between">
              <button
                onClick={handleGenerateActionItems}
                disabled={isGeneratingActions}
                className={`flex items-center px-4 py-2 ${
                  isGeneratingActions ? 'bg-gray-400' : 'bg-[#5236ab] hover:bg-[#4527a0]'
                } text-white rounded-md transition-colors`}
              >
                {Icon('ListChecks', 16, "w-4 h-4 mr-2")}
                {isGeneratingActions ? 'Extracting Actions...' : 'Extract Action Items'}
              </button>
              <button
                onClick={saveSummary}
                className="flex items-center px-4 py-2 bg-[#5236ab] text-white rounded-md hover:bg-[#4527a0] transition-colors"
              >
                {Icon('Save', 16, "mr-2")}
                Save Summary
              </button>
            </div>
          </Card>
        )}

        {/* Action Items Display */}
        {actionItems && (
          <Card title="Action Items" icon={Icon('CheckSquare')} fullWidth className="mb-4">
            <div className="space-y-4">
              {typeof actionItems === 'string' ? (
                <div className="bg-gray-50 p-4 rounded-lg">
                  {actionItems.includes('no specific action items') || actionItems.trim() === '' ? (
                    <div className="text-center py-4">
                      <div className="text-gray-500 flex items-center justify-center mb-2">
                        {Icon('Info', 24, "text-gray-400 mr-2")}
                      </div>
                      <p className="text-gray-600 font-medium">No action items found in this transcript.</p>
                    </div>
                  ) : (
                    <div dangerouslySetInnerHTML={{ __html: actionItems }}></div>
                  )}
                </div>
              ) : (
                <>
                  {Array.isArray(actionItems) && actionItems.map((item, index) => (
                    <div 
                      key={index}
                      className="bg-gray-50 p-4 rounded-lg flex items-start"
                    >
                      {Icon('CheckSquare', 20, "w-5 h-5 text-[#5236ab] mt-1 mr-3")}
                      <div>
                        <p className="font-medium text-gray-900">{item.task}</p>
                        <p className="text-gray-600">Assignee: {item.assignee}</p>
                      </div>
                    </div>
                  ))}
                </>
              )}
              <div className="flex justify-end mt-4">
                <button
                  onClick={saveActionItems}
                  className="flex items-center px-4 py-2 bg-[#5236ab] text-white rounded-md hover:bg-[#4527a0] transition-colors"
                >
                  {Icon('Save', 16, "w-4 h-4 mr-2")}
                  Save Action Items
                </button>
              </div>
            </div>
          </Card>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mt-4">
            <div className="flex items-center">
              {Icon('AlertCircle', 20, "w-5 h-5 mr-2")}
              {error}
            </div>
          </div>
        )}

        {/* Loading States */}
        {transcriber.progressItems.length > 0 && (
          <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 p-4 shadow-lg">
            <div className="max-w-7xl mx-auto">
              <div className="flex items-center mb-2">
                {Icon('Settings', 20, "h-5 w-5 text-[#5236ab] mr-2")}
                <p className="text-sm font-medium text-gray-800">Loading Model Files</p>
              </div>
              <div className="space-y-2">
                {transcriber.progressItems.map((item) => (
                  <Progress key={item.file} text={item.file} percentage={item.progress} />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Enhanced Processing Indicator */}
        {(transcriber.isBusy || isProcessing || isGeneratingActions || isTranscribing) && (
          <div className="fixed bottom-4 right-4 bg-white rounded-lg shadow-lg p-4 z-40 flex items-center">
            <div className="mr-3 w-6 h-6 border-t-2 border-[#5236ab] border-solid rounded-full animate-spin" />
            <div>
              <p className="font-medium text-gray-800 text-sm">
                {isTranscribing ? 'Generating Transcript...' :
                 isProcessing ? 'Generating Summary...' : 
                 isGeneratingActions ? 'Extracting Action Items...' : 
                 'Processing Audio...'}
              </p>
              <p className="text-xs text-gray-500">
                {isTranscribing ? 'Analyzing audio content...' :
                 isProcessing ? 'Creating meeting summary...' :
                 isGeneratingActions ? 'Finding actionable items...' :
                 'Processing in background...'}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;