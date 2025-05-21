import React, { useState, useRef } from 'react';
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

// No need for LucideIcon type with direct imports

const API_URL = 'http://localhost:8000';
const LIVE_CHUNK_INTERVAL_MS = 3000; // Interval for sending live audio chunks (in milliseconds)

interface ActionItem {
  task: string;
  assignee: string;
}

// Card props are defined in the Card component file

// Using imported Card component from './components/Card'

// StatusCard props and component are imported from './components/StatusCard'

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
  const [error, setError] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isScreenRecording, setIsScreenRecording] = useState(false);
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState('');
  const [liveSegments, setLiveSegments] = useState<Array<{text: string, start?: number, end?: number}>>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const screenMediaRecorderRef = useRef<MediaRecorder | null>(null);
  const screenAudioChunksRef = useRef<Blob[]>([]);
  const [isCppLiveActive, setIsCppLiveActive] = useState(false); // For whisper.cpp live transcription mode
  const totalLiveStreamDurationRef = useRef<number>(0); // Tracks cumulative duration for live transcript timestamps
  const audioContextRef = useRef<AudioContext | null>(null);
  const webSocketRef = useRef<WebSocket | null>(null);
  
  // Audio device state and fetching logic removed as we now use the default system audio device.

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
    if (transcriber.output && transcriber.output.text) {
      try {
        console.log('Starting summary generation...');
        setError('');
        setIsProcessing(true);
        
        console.log('Transcript length:', transcriber.output.text.length);
        const summaryText = await generateSummaryFromAPI(transcriber.output.text);
        
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
      console.warn('No transcript available for summary generation');
    }
  };

  // Action Items Generation - now uses direct response from Ollama API
  const generateActionItems = async (transcript: string) => {
    setIsGeneratingActions(true);
    setError('');
    
    try {
      // Make a request to the new API endpoint for extracting action items
      const response = await fetch(`${API_URL}/extract-action-items`, {
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
    if (!transcriber.output?.text) {
      setError('No transcript available to generate action items');
      return;
    }
    
    setIsGeneratingActions(true);
    setError('');
    
    try {
      const response = await axios.post<ActionItemType[] | string>(`${API_URL}/generate-action-items`, {
        text: transcriber.output.text
      });
      
      // Check if the response is an array of action items or HTML
      if (Array.isArray(response.data)) {
        setActionItems(response.data);
      } else if (typeof response.data === 'string') {
        setActionItems(response.data);
      } else {
        setError('Unexpected response format from server');
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
  
  const audioPlayerRef = useRef<HTMLAudioElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  // Helper function to convert HH:MM:SS.mmm to seconds
  const timeStringToSeconds = (timeStr: string): number => {
    if (!timeStr || typeof timeStr !== 'string') {
      console.warn(`Invalid time string input: ${timeStr}`);
      return 0;
    }
    const parts = timeStr.split(':');
    if (parts.length !== 3) {
      console.warn(`Invalid time string format: ${timeStr}`);
      return 0;
    }
    try {
      const h = parseInt(parts[0], 10);
      const m = parseInt(parts[1], 10);
      const sMs = parseFloat(parts[2]);
      if (isNaN(h) || isNaN(m) || isNaN(sMs)) {
        console.warn(`Non-numeric part in time string: ${timeStr}`);
        return 0;
      }
      return (h * 3600) + (m * 60) + sMs;
    } catch (e) {
      console.error(`Error parsing time string ${timeStr}:`, e);
      return 0;
    }
  };

  const startAudioRecording = (withLiveTranscript: boolean): void => {
    if (isRecording) {
      console.log('Already recording audio. Ignoring request.');
      return;
    }

    console.log(`Attempting to start audio recording with live transcript: ${withLiveTranscript}`);
    setError('');
    audioChunksRef.current = []; // Clear previous full recording chunks
    setLiveSegments([]);       // Clear previous live segments
    totalLiveStreamDurationRef.current = 0; // Reset live stream duration

    // Change this to always false - don't use whisper.cpp
    setIsCppLiveActive(false);
    
    if (withLiveTranscript) {
      // Initialize faster-whisper transcription
      console.log('Using faster-whisper for live transcription');
      transcriber.setEngine('faster_whisper');
      
      // Use WebSocket instead of chunk API
      startFasterWhisperWebSocket();
    }
    
    setIsLiveMode(withLiveTranscript);

    transcriber.output = undefined;
    setActionItems([]); 
    setSummary('');       
    setLiveTranscript(''); 
    setCurrentStep(1); 

    navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 }})
    .then(stream => {
      console.log('Microphone access granted.');
      const options = { mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: 128000 };
      let recorder;
      try {
        recorder = new MediaRecorder(stream, options);
      } catch (e) {
        console.error('Error creating MediaRecorder:', e);
        setError(`Failed to initialize recording: ${e instanceof Error ? e.message : 'Unknown error'}`);
        stream.getTracks().forEach(track => track.stop());
        setIsCppLiveActive(false); setIsLiveMode(false); setIsRecording(false);
        return;
      }
      mediaRecorderRef.current = recorder;

      mediaRecorderRef.current.ondataavailable = (event: BlobEvent) => {
        console.log('[App.tsx] ondataavailable called. Chunk size:', event.data?.size);
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data); // Collect full recording
          // Removed whisper.cpp endpoint call
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        console.log('MediaRecorder stopped');
        setIsRecording(false);
        
        const fullAudioBlob = new Blob(audioChunksRef.current, { type: mediaRecorderRef.current?.mimeType || 'audio/webm' });
        
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
      };

      try {
        if (withLiveTranscript) { 
          mediaRecorderRef.current.start(LIVE_CHUNK_INTERVAL_MS);
          console.log(`MediaRecorder started for live mode with ${LIVE_CHUNK_INTERVAL_MS}ms timeslice.`);
        } else {
          mediaRecorderRef.current.start();
          console.log('MediaRecorder started for standard recording.');
        }
        setIsRecording(true);
      } catch (e) {
        console.error('Error starting MediaRecorder:', e);
        setError(`Failed to start recording: ${e instanceof Error ? e.message : 'Unknown error'}`);
        stream.getTracks().forEach(track => track.stop());
        setIsRecording(false); setIsCppLiveActive(false); setIsLiveMode(false);
      }
    })
    .catch(err => {
      console.error('Failed to get user media (microphone):', err);
      setError('Could not access microphone. Please check permissions.');
      setIsRecording(false); setIsCppLiveActive(false); setIsLiveMode(false);
    });
  };

  const handleGenerateTranscript = async () => {
    if (!recordedBlob) {
      console.error('No recorded audio available for transcription.');
      setError('No recorded audio available. Please record audio first.');
      return;
    }

    setIsProcessing(true);
    setError('');

    try {
      console.log('Starting transcription of recorded audio...');
      
      // Create a file from the recorded blob
      const audioFile = new File([recordedBlob], 'recording.webm', { type: recordedBlob.type });
      
      // Call the transcriber's transcribeFile method
      if (transcriber && typeof transcriber.transcribeFile === 'function') {
        const result = await transcriber.transcribeFile(audioFile);
        
        if (result.error) {
          throw new Error(result.error);
        }
        
        console.log('Transcription successful:', result);
        
        // Update the UI with the transcription result
        if (result.transcript) {
          setLiveTranscript(result.transcript);
          setCurrentStep(3); // Move to the next step in the UI
        }
      } else {
        throw new Error('Transcriber not properly initialized');
      }
    } catch (err) {
      console.error('Error generating transcript:', err);
      setError(`Failed to generate transcript: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Add new function for WebSocket connection
  const startFasterWhisperWebSocket = async () => {
    try {
      // Start the transcription service
      const response = await axios.post(`${API_URL}/transcriber/start`, {
        model_name: "base",
        language: "en",
        engine: "faster_whisper"
      });
      
      // Connect to WebSocket
      const ws = new WebSocket(`ws://localhost:8000/transcriber/ws`);
      
      ws.onopen = () => {
        ws.send(JSON.stringify({
          engine: "faster_whisper",
          model_name: "base",
          language: "en"
        }));
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Received transcript from WebSocket:', data);
        
        if (data.segments) {
          setLiveSegments(prev => [...prev, ...data.segments.map((seg: { text: string; start?: number; end?: number }) => ({
            text: seg.text,
            start: seg.start,
            end: seg.end
          }))]);
        }
      };
    } catch (error) {
      console.error('Error starting faster-whisper WebSocket:', error);
    }
  };

  const startScreenRecording = async (withLiveTranscript: boolean) => {
    if (isScreenRecording) {
      console.log('Already screen recording.');
      return;
    }
    
    console.log(`Starting screen recording. Live transcript: ${withLiveTranscript}, Engine: ${transcriber.engine}`);
    setIsScreenRecording(true);
    setError('');
    screenAudioChunksRef.current = [];
    setLiveSegments([]);
    
    // Use faster-whisper for screen recording as well
    transcriber.setEngine('faster-whisper');
    setIsLiveMode(withLiveTranscript);

    console.log('Starting screen recording. Live transcript:', withLiveTranscript);

    try {
      const displayStream = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: 'monitor', frameRate: { ideal: 30 }, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: true // Capture system audio
      });

      const audioStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
      });

      const combinedStream = new MediaStream();
      displayStream.getTracks().forEach(track => combinedStream.addTrack(track));
      audioStream.getTracks().forEach(track => combinedStream.addTrack(track));

      console.log('Combined stream for screen recording created. Video:', combinedStream.getVideoTracks().length, 'Audio:', combinedStream.getAudioTracks().length);

      const recorderOptions = {
        mimeType: 'video/webm;codecs=vp9,opus',
        videoBitsPerSecond: 3000000, // 3 Mbps for video
        audioBitsPerSecond: 128000,  // 128 kbps for audio
      };
      const screenRecorder = new MediaRecorder(combinedStream, recorderOptions);
      screenMediaRecorderRef.current = screenRecorder;

      screenRecorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          screenAudioChunksRef.current.push(event.data);
        }
      };

      screenRecorder.onstop = () => {
        console.log('Screen recorder stopped.');
        setIsScreenRecording(false);
        const videoBlob = new Blob(screenAudioChunksRef.current, { type: 'video/webm' });
        if (videoBlob.size > 0) {
          const videoFile = new File([videoBlob], 'screen_recording.webm', { type: videoBlob.type });
          console.log('Screen recording finished, blob size:', videoBlob.size);
          
          // Send the recorded video for transcription
          if (transcriber && typeof transcriber.transcribeFile === 'function') {
            transcriber.transcribeFile(videoFile);
          }
        }
      };

      console.log('MediaRecorder started');
      
      // For live mode with screen recording
      if (withLiveTranscript) {
        console.log('Live transcription enabled for screen recording');
        
        // Create an audio context to extract audio from the screen capture
        const audioContext = new AudioContext();
        audioContextRef.current = audioContext;
        
        // Create a MediaStreamAudioSourceNode
        // Feed the audio from our combined stream into the audio context
        const audioSource = audioContext.createMediaStreamSource(combinedStream);
        
        // Create a processor to send audio data to transcription service
        const processor = audioContext.createScriptProcessor(4096, 1, 1);
        
        processor.onaudioprocess = async (e) => {
          // Get audio data
          const audioData = e.inputBuffer.getChannelData(0);
          
          // Convert to format suitable for transcription (16-bit PCM)
          const pcmData = Int16Array.from(audioData.map(x => x * 0x7FFF));
          
          // Send to WebSocket if connected
          if (webSocketRef.current?.readyState === WebSocket.OPEN) {
            webSocketRef.current.send(pcmData.buffer);
          }
        };
        
        // Connect the nodes
        audioSource.connect(processor);
        processor.connect(audioContext.destination);
        
        // Initialize WebSocket for transcription
        const ws = new WebSocket(`ws://localhost:8000/transcriber/ws`);
        webSocketRef.current = ws;
        
        ws.onopen = () => {
          console.log('WebSocket open for screen recording transcription');
          ws.send(JSON.stringify({
            engine: 'faster-whisper',
            model_name: transcriber.model,
            language: transcriber.language || 'en'
          }));
        };
        
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('Screen recording transcription received:', data);
            
            if (data.segments && data.segments.length > 0) {
              setLiveSegments(prev => [...prev, ...data.segments]);
            }
          } catch (err) {
            console.error('Error processing transcript:', err);
          }
        };
      }

      // Start the screen recorder
      screenRecorder.start(1000); // Collect data every second
      
    } catch (err) {
      console.error('Error starting screen recording:', err);
      setError('Failed to start screen recording. Check permissions.');
      setIsScreenRecording(false);
    }
  };

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
    }
  };

  const stopScreenRecording = (): void => {
    console.log('stopScreenRecording called. Current state:', screenMediaRecorderRef.current?.state);
    if (screenMediaRecorderRef.current && screenMediaRecorderRef.current.state === 'recording') {
      screenMediaRecorderRef.current.stop(); // Triggers 'onstop'
    } else {
      console.warn('Screen recorder not recording or not initialized.');
      setIsScreenRecording(false);
    }
    // Stream tracks are stopped in onstop or onerror.
    
    // Also close audio context if it exists
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(e => console.error('Error closing audio context:', e));
      audioContextRef.current = null;
    }
    
    // Close WebSocket connection
    if (webSocketRef.current) {
      webSocketRef.current.close();
      webSocketRef.current = null;
    }
    
    // Add code to update transcript state with live segments
    if (liveSegments.length > 0) {
      const fullText = liveSegments.map(segment => segment.text).join(' ');
      
      transcriber.output = {
        isBusy: false,
        text: fullText,
        chunks: liveSegments.map(segment => ({
          text: segment.text,
          timestamp: [segment.start || 0, segment.end || 0]
        }))
      };
      
      setCurrentStep(2);
    }
  };

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
        
        {/* Audio device configuration section removed - now using system default device */}
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
            value={transcriber.output ? "Active" : "Waiting"}
          />
        </div>

        {/* Audio Input Options */}
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
                        transcriber.transcribeFile(file);
                      }
                    }}
                  />
                </button>

                {/* Audio Recording Button */}
                <button
                  className="flex flex-col items-center justify-center p-6 border-2 border-gray-200 rounded-xl hover:border-[#5236ab] hover:bg-gray-50 transition-all duration-200"
                  onClick={() => {
                    setIsRecording(true);
                    if (isLiveMode) {
                      startAudioRecording(true);
                    } else {
                      startAudioRecording(false);
                    }
                  }}
                >
                  {Icon('Mic', 32, "w-8 h-8 mb-2 text-[#5236ab]")}
                  <span className="font-medium">Audio Recording</span>
                  <span className="text-xs text-gray-500 mt-1">{isLiveMode ? 'With Live Transcript' : 'Standard'}</span>
                </button>

                {/* Screen Recording Button */}
                <button
                  className="flex flex-col items-center justify-center p-6 border-2 border-gray-200 rounded-xl hover:border-[#5236ab] hover:bg-gray-50 transition-all duration-200"
                  onClick={() => {
                    setIsScreenRecording(true);
                    if (isLiveMode) {
                      startScreenRecording(true);
                    } else {
                      startScreenRecording(false);
                    }
                  }}
                >
                  {Icon('Monitor', 32, "w-8 h-8 mb-2 text-[#5236ab]")}
                  <span className="font-medium">Screen Recording</span>
                  <span className="text-xs text-gray-500 mt-1">{isLiveMode ? 'With Live Transcript' : 'Standard'}</span>
                </button>
              </div>
            )}
            {/* Stop Recording Button when recording is active */}
            {(isRecording || isScreenRecording) && (
              <div className="flex justify-center">
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
              </div>
            )}
            
            {/* Recording status */}
            {(isRecording || isScreenRecording) && (
              <div className="flex items-center justify-center mt-2">
                <div className="w-3 h-3 bg-red-500 rounded-full mr-2 animate-pulse"></div>
                <span className="text-sm font-medium">
                  {isRecording ? 'Recording audio...' : 'Recording screen and audio...'}
                  {isLiveMode && ' with live transcription'}
                </span>
              </div>
            )}
            
            {/* Audio player for recorded audio */}
            {audioUrl && (
          <div className="mt-4 flex justify-center">
            <button
              onClick={handleGenerateTranscript}
              className="bg-[#5236ab] hover:bg-[#4527a0] text-white font-medium py-2 px-6 rounded-lg shadow-md transition-colors duration-200 flex items-center"
              disabled={isProcessing} // Disable button when processing
            >
              {Icon('FileText', 20, "w-5 h-5 mr-2")} 
              {isProcessing ? 'Generating Transcript...' : 'Generate Transcript'}
            </button>
          </div>
        )}

        {/* Original audio player part - assuming it was after the part to be replaced or this is a new addition point*/}
        {audioUrl && (
              <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                <p className="font-medium text-gray-700 mb-2">Recorded Audio:</p>
                <audio ref={audioPlayerRef} controls className="w-full" src={audioUrl}></audio>
              </div>
            )}
          </div>
        </Card>

        {/* Live Transcript Display */}
        {((isRecording || isScreenRecording) && isLiveMode) && (
          <Card title="Live Transcript with Timestamps" icon={Icon('Mic')} fullWidth>
            <div className="max-h-80 overflow-y-auto my-2 p-4 bg-gray-50 rounded-lg">
              {/* Debug info */}
              <div className="mb-2 p-2 bg-gray-100 text-xs rounded">
                <div className="grid grid-cols-2 gap-1">
                  <div>Live segments: <span className="font-mono">{liveSegments.length || 0}</span></div>
                  <div>Text length: <span className="font-mono">{liveTranscript?.length || 0}</span></div>
                  <div>Recording: <span className="font-mono">{isRecording ? 'Yes' : 'No'}</span></div>
                  <div>Live mode: <span className="font-mono">{isLiveMode ? 'Yes' : 'No'}</span></div>
                  <div>Engine: <span className="font-mono">{isLiveMode && (isRecording || isScreenRecording) ? 'whisper.cpp (live)' : transcriber.engine}</span></div>
                </div>
              </div>
              
              {liveSegments.length > 0 ? (
                <div className="space-y-2">
                  {liveSegments.map((segment, index) => (
                    <div key={index} className="border-b border-gray-200 pb-2 last:border-0">
                      <div className="text-sm text-gray-500 mb-1">
                        {segment.start !== undefined ? formatTimestamp(segment.start) : '00:00:00'} 
                        {segment.end !== undefined ? ` - ${formatTimestamp(segment.end)}` : ''}
                      </div>
                      <div className="text-gray-800">{segment.text}</div>
                    </div>
                  ))}
                </div>
              ) : liveTranscript ? (
                <div className="whitespace-pre-wrap text-gray-800">{liveTranscript}</div>
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

        {/* Transcript Display - with extra debug information */}
        <div className="mb-4">
          <pre className="bg-gray-100 p-2 text-xs">
            Transcript State: {transcriber.output ? 'Has Data' : 'No Data'}
            {transcriber.output && `, Text Length: ${transcriber.output.text?.length || 0}`}
            {transcriber.output && `, Chunks: ${transcriber.output.chunks?.length || 0}`}
          </pre>
        </div>
        
        {/* Always display the transcript card, even if empty */}
        <Card title="Transcript" icon={Icon('FileText')} fullWidth>
          {transcriber.output ? (
            <>
              <Transcript transcribedData={transcriber.output} />
              <div className="mt-4 flex justify-end space-x-4">
                <button
                  onClick={handleGenerateSummary}
                  disabled={isProcessing}
                  className={`flex items-center px-4 py-2 ${
                    isProcessing ? 'bg-gray-400' : 'bg-[#5236ab] hover:bg-[#4527a0]'
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
          <Card title="Meeting Summary" icon={Icon('FileText')} fullWidth>
            <div className="bg-gray-50 p-4 rounded-lg mb-4">
              <div className="prose max-w-none">
                {/* Use dangerouslySetInnerHTML to render HTML content safely */}
                <div dangerouslySetInnerHTML={{ __html: summary }}></div>
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

        {/* Action Items Display - Either from parsed actionItems array or raw HTML */}
        {actionItems && (
          <Card title="Action Items" icon={Icon('CheckSquare')} fullWidth>
            <div className="space-y-4">
              {typeof actionItems === 'string' ? (
                // Display HTML action items
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div dangerouslySetInnerHTML={{ __html: actionItems }}></div>
                </div>
              ) : (
                // Display traditional action items array
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
            {error}
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

        {/* Processing Indicator */}
        {(transcriber.isBusy || isProcessing || isGeneratingActions) && (
          <div className="fixed bottom-4 right-4 bg-white rounded-lg shadow-lg p-4 z-40 flex items-center">
            <div className="mr-3 w-6 h-6 border-t-2 border-[#5236ab] border-solid rounded-full animate-spin" />
            <div>
              <p className="font-medium text-gray-800 text-sm">
                {isProcessing ? 'Generating Summary' : 
                 isGeneratingActions ? 'Extracting Action Items' : 
                 'Processing Audio'}
              </p>
              <p className="text-xs text-gray-500">Processing in background...</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
