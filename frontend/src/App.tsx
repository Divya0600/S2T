import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useTranscriber, formatTimestamp } from "./hooks/useTranscriber";
import Transcript from "./components/Transcript";
import Progress from "./components/Progress";
import Card from "./components/Card";
import StatusCard from "./components/StatusCard";
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import * as LucideIcons from 'lucide-react';

const Icon = (icon: keyof typeof LucideIcons, size = 20, className = '') => {
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
  
  type TranscriptSegment = {
    text: string, 
    start?: number, 
    end?: number, 
    timestamp?: [number, number | null],
    id?: string
  };
  
  const [liveSegments, setLiveSegments] = useState<TranscriptSegment[]>([]);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const screenMediaRecorderRef = useRef<MediaRecorder | null>(null);
  const screenAudioChunksRef = useRef<Blob[]>([]);
  const [uploadedAudioFile, setUploadedAudioFile] = useState<File | null>(null);
  
  const [systemAudioAvailable, setSystemAudioAvailable] = useState<boolean>(false);
  const [audioStreamType, setAudioStreamType] = useState<'mic-only' | 'system-only' | 'combined'>('combined');
  const [currentAudioDevice, setCurrentAudioDevice] = useState<string>('Default');
  const [availableAudioDevices, setAvailableAudioDevices] = useState<MediaDeviceInfo[]>([]);
  
  const webSocketRef = useRef<WebSocket | null>(null);
  const webSocketSessionId = useRef<string | null>(null);
  const isLiveTranscriptionActive = useRef<boolean>(false);
  const processedSegmentIds = useRef<Set<string>>(new Set());

  const audioContextRef = useRef<AudioContext | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const systemStreamRef = useRef<MediaStream | null>(null);
  const combinedStreamRef = useRef<MediaStream | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  
  const audioProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const lastAudioSendTime = useRef<number>(0);
  const recordingStartTime = useRef<number>(0);
  const audioWorkletRef = useRef<AudioWorkletNode | null>(null);
  const selectedDeviceId = useRef<string | null>(null);

  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then(devices => {
      const audioInputs = devices.filter(device => device.kind === 'audioinput');
      setAvailableAudioDevices(audioInputs);
      if (audioInputs.length > 0) {
        const defaultDevice = audioInputs.find(d => d.deviceId === 'default') || audioInputs[0];
        setCurrentAudioDevice(defaultDevice.label || 'Default');
      }
    });

    navigator.mediaDevices.addEventListener('devicechange', handleDeviceChange);
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', handleDeviceChange);
    };
  }, []);

  const handleDeviceChange = useCallback(async () => {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const audioInputs = devices.filter(device => device.kind === 'audioinput');
    setAvailableAudioDevices(audioInputs);
    
    if (isRecording || isScreenRecording) {
      await updateAudioStream();
    }
  }, [isRecording, isScreenRecording]);

  const updateAudioStream = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices.filter(device => device.kind === 'audioinput');
      
      let newStream: MediaStream;
      if (selectedDeviceId.current && audioInputs.find(d => d.deviceId === selectedDeviceId.current)) {
        newStream = await navigator.mediaDevices.getUserMedia({
          audio: { 
            deviceId: { exact: selectedDeviceId.current },
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            sampleRate: 48000
          }
        });
      } else {
        newStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            sampleRate: 48000
          }
        });
      }

      if (micStreamRef.current) {
        micStreamRef.current.getTracks().forEach(track => track.stop());
      }
      micStreamRef.current = newStream;

      if (isLiveMode && audioProcessorRef.current) {
        setupLiveAudioProcessing(newStream, true);
      }

      const newDevice = audioInputs.find(d => d.deviceId === newStream.getAudioTracks()[0].getSettings().deviceId);
      if (newDevice) {
        setCurrentAudioDevice(newDevice.label || 'Default');
      }
    } catch (error) {
      console.error('Error updating audio stream:', error);
    }
  }, [isLiveMode]);

  const generateSummaryFromAPI = async (transcript: string) => {
    try {
      const response = await fetch(`${API_URL}/summarize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ transcript }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate summary');
      }

      const data = await response.json();
      
      if (data.status === 'error') {
        throw new Error(data.detail || 'Failed to generate summary');
      }
      
      if (data.status === 'completed') {
        return data.summary;
      } else {
        throw new Error('Unexpected response from server');
      }
    } catch (error) {
      throw error;
    }
  };

  const handleGenerateSummary = async () => {
    if (liveTranscript && liveTranscript.trim() !== '') {
      try {
        setError('');
        setIsProcessing(true);
        
        const summaryText = await generateSummaryFromAPI(liveTranscript);
        
        setSummary(summaryText);
        setCurrentStep(4);
      } catch (error) {
        const errorMessage = 'Failed to generate summary. Please try again.';
        setError(errorMessage);
      } finally {
        setIsProcessing(false);
      }
    } else {
      setError('No transcript available for summary generation.');
    }
  };

  const generateActionItems = async (transcript: string) => {
    setIsGeneratingActions(true);
    setError('');
    
    try {
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
      
      if (result.status === 'error') {
        setError(`Error extracting action items: ${result.detail}`);
        setIsGeneratingActions(false);
        return;
      }
      
      if (result.status === 'completed') {
        const actionItemsText = result.action_items;
        
        try {
          const result = parseActionItems(actionItemsText);
          if (result && result.length > 0) {
            setActionItems(result);
          } else {
            setActionItems(actionItemsText || '');
          }
        } catch (e) {
          setActionItems(actionItemsText || '');
        }
        
        setCurrentStep(5);
        
        setIsGeneratingActions(false);
        return;
      }
      
      setError(`Unknown status received: ${result.status}`);
    } catch (error) {
      setError(`Failed to extract action items: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsGeneratingActions(false);
    }
  };

  const handleGenerateActionItems = async () => {
    if (!liveTranscript || liveTranscript.trim() === '') {
      setError('No transcript available to generate action items');
      return;
    }
    
    setIsGeneratingActions(true);
    setError('');
    
    try {
      const response = await axios.post<{ status: string; action_items?: ActionItemType[] | string; detail?: string }>(`${API_URL}/generate-action-items`, {
        transcript: liveTranscript
      });
      
      if (response.data && response.data.status === 'completed' && 'action_items' in response.data) {
        const actionItemsData = response.data.action_items;
        if (Array.isArray(actionItemsData)) {
          setActionItems(actionItemsData);
        } else if (typeof actionItemsData === 'string') {
          setActionItems(actionItemsData);
        } else {
          setError('Action items data received, but not in expected format (array or string).');
          setActionItems(null);
        }
      } else if (response.data && response.data.status === 'error' && response.data.detail) {
        setError(response.data.detail);
        setActionItems(null);
      } else {
        setError('Unexpected response format from server.');
        setActionItems(null);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setError(`Failed to generate action items: ${errorMessage}`);
    } finally {
      setIsGeneratingActions(false);
    }
  };

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

  const formatMarkdown = (text: string) => {
    return text;
  };

  const createAudioContext = useCallback(() => {
    if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 48000 });
    }
    return audioContextRef.current;
  }, []);

  const combineAudioStreams = useCallback(async (micStream: MediaStream | null, systemStream: MediaStream | null): Promise<MediaStream> => {
    if (!micStream && !systemStream) {
      throw new Error('No audio streams available to combine');
    }

    if (!micStream && systemStream) {
      setAudioStreamType('system-only');
      return systemStream;
    }
    
    if (micStream && !systemStream) {
      setAudioStreamType('mic-only');
      return micStream;
    }

    try {
      const audioContext = createAudioContext();
      
      const destination = audioContext.createMediaStreamDestination();
      
      if (micStream && micStream.getAudioTracks().length > 0) {
        const micSource = audioContext.createMediaStreamSource(micStream);
        const micGain = audioContext.createGain();
        micGain.gain.value = 1.0;
        micSource.connect(micGain);
        micGain.connect(destination);
      }
      
      if (systemStream && systemStream.getAudioTracks().length > 0) {
        const systemSource = audioContext.createMediaStreamSource(systemStream);
        const systemGain = audioContext.createGain();
        systemGain.gain.value = 1.2;
        systemSource.connect(systemGain);
        systemGain.connect(destination);
      }
      
      setAudioStreamType('combined');
      return destination.stream;
      
    } catch (error) {
      setAudioStreamType('mic-only');
      return micStream!;
    }
  }, [createAudioContext]);

  const initializeWebSocketConnection = useCallback(() => {
    if (webSocketRef.current && webSocketRef.current.readyState === WebSocket.OPEN) {
      return;
    }

    if (webSocketRef.current) {
      cleanupWebSocket();
    }
    
    const wsUrl = `ws://localhost:8000/transcriber/ws?engine=faster-whisper`;
    const ws = new WebSocket(wsUrl);
    webSocketRef.current = ws;
    isLiveTranscriptionActive.current = true;
    
    processedSegmentIds.current.clear();
    
    ws.onopen = () => {
      ws.send(JSON.stringify({
        engine: 'faster_whisper',
        model_name: 'base',
        language: 'en'
      }));
    };
    
    ws.onmessage = (event) => {
      if (!isLiveTranscriptionActive.current) {
        return;
      }

      try {
        const data = JSON.parse(event.data);
        
        if (data.heartbeat) {
          return;
        }
        
        if (data.session_id) {
          webSocketSessionId.current = data.session_id;
        }
        
        if (data.segments && Array.isArray(data.segments)) {
          const newValidSegments: TranscriptSegment[] = [];
          
          for (const segment of data.segments) {
            const segmentText = segment.text?.trim();
            const segmentStart = segment.start || 0;
            const segmentEnd = segment.end || segmentStart;
            
            if (!segmentText || segmentText.length < 3) {
              continue;
            }
            
            const hallucinations = [
              'thanks for watching', 'thank you for watching', 
              'bye', 'goodbye', 'see you later',
              'subscribe', 'like and subscribe',
              'thanks', 'thank you', 'music',
              'applause', '[music]', '[applause]',
              '♪', 'mm', 'hmm', 'uh', 'um',
              'you', 'the', 'and', 'a'
            ];
            
            if (hallucinations.includes(segmentText.toLowerCase()) || 
                segmentText.length < 3 || 
                /^[♪\[\]()]+$/.test(segmentText)) {
              continue;
            }
            
            const segmentId = `${segmentText}_${segmentStart.toFixed(2)}_${segmentEnd.toFixed(2)}`;
            
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
          
          if (newValidSegments.length > 0) {
            setLiveSegments(prevSegments => {
              const updatedSegments = [...prevSegments, ...newValidSegments];
              
              updatedSegments.sort((a, b) => (a.start || 0) - (b.start || 0));
              
              const fullText = updatedSegments.map(seg => seg.text).join(' ');
              setLiveTranscript(fullText);
              
              return updatedSegments;
            });
          }
        }
      } catch (error) {
        console.error('Error processing WebSocket message:', error);
      }
    };
    
    ws.onerror = (error) => {
      setError('Connection error during live transcription');
    };
    
    ws.onclose = (event) => {
      if (isLiveTranscriptionActive.current) {
        cleanupWebSocket();
      }
    };
  }, []);

  const cleanupWebSocket = useCallback(() => {
    isLiveTranscriptionActive.current = false;
    
    if (webSocketRef.current) {
      const ws = webSocketRef.current;
      
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      
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
  }, []);

  const cleanupAudioResources = useCallback(() => {
    if (audioProcessorRef.current) {
      audioProcessorRef.current.disconnect();
      audioProcessorRef.current = null;
    }
    
    if (audioWorkletRef.current) {
      audioWorkletRef.current.disconnect();
      audioWorkletRef.current = null;
    }
    
    [micStreamRef, systemStreamRef, combinedStreamRef, streamRef].forEach(streamRef => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          track.stop();
          track.enabled = false;
        });
        streamRef.current = null;
      }
    });
    
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(console.error);
      audioContextRef.current = null;
    }
    
    setSystemAudioAvailable(false);
    setAudioStreamType('combined');
  }, []);

  useEffect(() => {
    return () => {
      cleanupWebSocket();
      cleanupAudioResources();
    };
  }, [cleanupWebSocket, cleanupAudioResources]);

  const sendAudioToWebSocket = useCallback((audioData: Float32Array, currentTime?: number) => {
    if (!webSocketRef.current || webSocketRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    try {
      const audioLevel = Math.sqrt(audioData.reduce((sum, x) => sum + x * x, 0) / audioData.length);
      if (audioLevel < 0.002) {
        return;
      }

      const targetSampleRate = 16000;
      const sourceSampleRate = audioContextRef.current?.sampleRate || 48000;
      let processedAudio = audioData;
      
      if (sourceSampleRate !== targetSampleRate) {
        const ratio = sourceSampleRate / targetSampleRate;
        const newLength = Math.floor(audioData.length / ratio);
        const downsampled = new Float32Array(newLength);
        
        for (let i = 0; i < newLength; i++) {
          const sourceIndex = Math.floor(i * ratio);
          downsampled[i] = audioData[sourceIndex];
        }
        processedAudio = downsampled;
      }

      const message = {
        type: 'audio_data',
        audio: Array.from(processedAudio),
        sample_rate: targetSampleRate,
        timestamp: currentTime || Date.now()
      };

      webSocketRef.current.send(JSON.stringify(message));
      
    } catch (error) {
      console.error('Error sending audio to WebSocket:', error);
    }
  }, []);

  const setupLiveAudioProcessing = useCallback((audioStream: MediaStream, isRecording = false) => {
    if (!audioStream || audioStream.getAudioTracks().length === 0) {
      return;
    }

    try {
      const audioContext = createAudioContext();
      
      if (audioContext.state === 'suspended') {
        audioContext.resume();
      }

      const source = audioContext.createMediaStreamSource(audioStream);
      
      const bufferSize = 16384;
      const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
      audioProcessorRef.current = processor;
      
      let audioBuffer: Float32Array[] = [];
      let bufferDuration = 0;
      const targetDuration = 1;
      const sampleRate = audioContext.sampleRate;

      processor.onaudioprocess = (event) => {
        if (!isLiveTranscriptionActive.current || (!isRecording && !isScreenRecording)) {
          return;
        }

        const inputBuffer = event.inputBuffer;
        const inputData = inputBuffer.getChannelData(0);
        
        const chunk = new Float32Array(inputData.length);
        chunk.set(inputData);
        audioBuffer.push(chunk);
        
        bufferDuration += inputData.length / sampleRate;
        
        if (bufferDuration >= targetDuration) {
          const totalLength = audioBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
          const combinedAudio = new Float32Array(totalLength);
          
          let offset = 0;
          for (const chunk of audioBuffer) {
            combinedAudio.set(chunk, offset);
            offset += chunk.length;
          }
          
          const currentTime = (Date.now() - recordingStartTime.current) / 1000;
          
          sendAudioToWebSocket(combinedAudio, currentTime);
          
          audioBuffer = [];
          bufferDuration = 0;
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);
      
    } catch (error) {
      console.error('Error setting up live audio processing:', error);
    }
  }, [createAudioContext, sendAudioToWebSocket, isScreenRecording]);

  const startAudioRecording = (withLiveTranscript: boolean): void => {
    if (isRecording) {
      return;
    }

    setError('');
    
    audioChunksRef.current = [];
    setLiveSegments([]);
    setLiveTranscript('');
    processedSegmentIds.current.clear();
    recordingStartTime.current = Date.now();
    
    setIsLiveMode(withLiveTranscript);

    transcriber.output = undefined;
    setActionItems([]); 
    setSummary('');       
    setCurrentStep(1); 

    if (withLiveTranscript) {
      initializeWebSocketConnection();
      
      setTimeout(() => {
        if (webSocketRef.current && webSocketRef.current.readyState === WebSocket.OPEN) {
          console.log('WebSocket connected, proceeding with audio setup');
        } else {
          console.warn('WebSocket not fully connected yet');
        }
      }, 500);
    }

    navigator.mediaDevices.getUserMedia({ 
      audio: { 
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        sampleRate: 48000,
        channelCount: 1 
      }
    })
    .then(stream => {
      streamRef.current = stream;
      micStreamRef.current = stream;
      
      if (withLiveTranscript) {
        setTimeout(() => {
          setupLiveAudioProcessing(stream, true);
        }, 1000);
      }
      
      const options = { 
        mimeType: 'audio/webm;codecs=opus', 
        audioBitsPerSecond: 128000 
      };
      
      let recorder;
      try {
        recorder = new MediaRecorder(stream, options);
      } catch (e) {
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

      mediaRecorderRef.current.ondataavailable = (event: BlobEvent) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        setIsRecording(false);
        
        const fullAudioBlob = new Blob(audioChunksRef.current, { 
          type: mediaRecorderRef.current?.mimeType || 'audio/webm' 
        });
        
        if (fullAudioBlob.size > 0) {
          const url = URL.createObjectURL(fullAudioBlob);
          setAudioUrl(url);
          
          setRecordedBlob(fullAudioBlob);
          
          setCurrentStep(2);
        }
        
        cleanupAudioResources();
        
        if (withLiveTranscript) {
          cleanupWebSocket();
        }
      };

      try {
        if (withLiveTranscript) { 
          mediaRecorderRef.current.start(500);
        } else {
          mediaRecorderRef.current.start();
        }
        setIsRecording(true);
      } catch (e) {
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
      setError('Could not access microphone. Please check permissions.');
      setIsRecording(false);
      setIsLiveMode(false);
    });
  };

  const handleGenerateTranscript = async (uploadedFile?: File) => {
    const audioToTranscribe = uploadedFile || recordedBlob;

    if (!audioToTranscribe) {
      setError('No audio available. Please record or upload audio first.');
      return;
    }

    setLiveSegments([]);
    setLiveTranscript('');
    processedSegmentIds.current.clear();

    if (audioToTranscribe instanceof Blob) {
      setAudioUrl(URL.createObjectURL(audioToTranscribe));
    }

    setIsTranscribing(true);
    setError('');

    try {
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
      
      if (result.status === 'completed') {
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
      setError(`Failed to generate transcript: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsTranscribing(false);
    }
  };

  const startScreenRecording = async (withLiveTranscript: boolean) => {
    if (isScreenRecording) {
      return;
    }
    
    setIsScreenRecording(true);
    setError('');
    
    screenAudioChunksRef.current = [];
    setLiveSegments([]);
    setLiveTranscript('');
    processedSegmentIds.current.clear();
    recordingStartTime.current = Date.now();
    cleanupAudioResources();
    
    setIsLiveMode(withLiveTranscript);

    try {
      const displayStream = await navigator.mediaDevices.getDisplayMedia({
        video: false,
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        }
      });

      const systemAudioTracks = displayStream.getAudioTracks();

      if (systemAudioTracks.length > 0) {
        setSystemAudioAvailable(true);
        systemStreamRef.current = new MediaStream(systemAudioTracks);
      } else {
        setSystemAudioAvailable(false);
        
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
        
        alert(`System audio not detected!\n\n${instruction}\n\nWe'll continue with microphone audio only.`);
      }

      let micStream: MediaStream | null = null;
      try {
        micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            sampleRate: 48000,
            channelCount: 1,
          }
        });
        
        micStreamRef.current = micStream;
      } catch (micError) {
        alert('Could not access microphone. Continuing with system audio only (if available).');
      }

      const finalAudioStream = await combineAudioStreams(micStream, systemStreamRef.current);
      
      combinedStreamRef.current = finalAudioStream;

      if (withLiveTranscript) {
        initializeWebSocketConnection();
        
        setTimeout(() => {
          setupLiveAudioProcessing(finalAudioStream, true);
        }, 1000);
      }

      const recorderOptions = {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 128000,
      };
      
      let screenRecorder: MediaRecorder;
      try {
        screenRecorder = new MediaRecorder(finalAudioStream, recorderOptions);
      } catch (e) {
        screenRecorder = new MediaRecorder(finalAudioStream);
      }
      
      screenMediaRecorderRef.current = screenRecorder;

      screenRecorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          screenAudioChunksRef.current.push(event.data);
        }
      };

      screenRecorder.onstop = async () => {
        setIsScreenRecording(false);
        
        const audioBlob = new Blob(screenAudioChunksRef.current, { 
          type: screenRecorder.mimeType || 'audio/webm' 
        });
        
        if (audioBlob.size > 0) {
          const url = URL.createObjectURL(audioBlob);
          setAudioUrl(url);
          
          setRecordedBlob(audioBlob);
          
          if (withLiveTranscript && liveSegments.length > 0) {
            setCurrentStep(3);
          } else {
            setCurrentStep(2);
          }
        } else {
          setError('Recording failed: No data captured');
        }
        
        cleanupAudioResources();
        
        if (withLiveTranscript) {
          cleanupWebSocket();
        }
      };

      try {
        screenRecorder.start(500);
        
        const audioInfo = systemAudioAvailable 
          ? (micStream ? 'System + Microphone Audio' : 'System Audio Only')
          : 'Microphone Audio Only';
        
      } catch (startError) {
        throw startError;
      }
      
    } catch (err) {
      let errorMessage = 'Failed to start recording. ';
      
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
      
      cleanupAudioResources();
      if (withLiveTranscript) {
        cleanupWebSocket();
      }
    }
  };

  const stopScreenRecording = useCallback(() => {
    setIsScreenRecording(false);
    
    if (screenMediaRecorderRef.current) {
      if (screenMediaRecorderRef.current.state === 'recording') {
        screenMediaRecorderRef.current.stop();
      }
      screenMediaRecorderRef.current = null;
    }
    
    cleanupAudioResources();
    
    cleanupWebSocket();
    
    setIsLiveMode(false);
  }, [cleanupWebSocket, cleanupAudioResources]);

  const stopAudioRecording = (): void => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    } else {
      setIsRecording(false);
      setIsLiveMode(false);
      
      cleanupAudioResources();
      
      cleanupWebSocket();
    }
  };

  useEffect(() => {
    if (liveSegments.length > 0 && isLiveMode) {
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
      const tempDiv = document.createElement('div');
      tempDiv.innerHTML = actionItems;
      content = tempDiv.textContent || tempDiv.innerText || '';
    } else if (Array.isArray(actionItems)) {
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

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <ProgressSteps currentStep={currentStep} />

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

        <Card title="Input Options" icon={Icon('Mic')} fullWidth>
          <div className="space-y-4">
            <div className="flex items-center justify-end mb-2">
              <span className="mr-2 text-sm font-medium text-gray-700">Live Mode</span>
              <button 
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${isLiveMode ? 'bg-[#5236ab]' : 'bg-gray-300'}`}
                onClick={() => setIsLiveMode(!isLiveMode)}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${isLiveMode ? 'translate-x-6' : 'translate-x-1'}`} />
              </button>
            </div>

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

            {!isRecording && !isScreenRecording && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
                        setRecordedBlob(null);
                        setLiveSegments([]);
                        setLiveTranscript('');
                        transcriber.output = undefined;
                        setActionItems([]);
                        setSummary('');
                        setCurrentStep(2);
                      }
                    }}
                  />
                </button>

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

                <button
                  className="flex flex-col items-center justify-center p-6 border-2 border-gray-200 rounded-xl hover:border-[#5236ab] hover:bg-gray-50 transition-all duration-200"
                  onClick={() => {
                    setIsScreenRecording(true);
                    startScreenRecording(isLiveMode);
                  }}
                >
                  {Icon('Monitor', 32, "w-8 h-8 mb-2 text-[#5236ab]")}
                  <span className="font-medium">System Audio Recording</span>
                  <span className="text-xs text-gray-500 mt-1">
                    {isLiveMode ? 'Live Teams/Meeting Audio' : 'Record Meeting Audio'}
                  </span>
                  <span className="text-xs text-blue-600 mt-1 font-medium">
                    Audio Only (No Video)
                  </span>
                </button>
              </div>
            )}
            
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
                  Stop {isScreenRecording ? 'Recording' : 'Audio Recording'}
                </button>
                
                <div className="text-center space-y-2">
                  <div className="flex items-center justify-center">
                    <div className="w-3 h-3 bg-red-500 rounded-full mr-2 animate-pulse"></div>
                    <span className="text-sm font-medium">
                      {isRecording ? 'Recording audio...' : 'Recording audio...'}
                      {isLiveMode && ' with live transcription'}
                    </span>
                  </div>
                  
                  <div className="text-xs text-gray-600 bg-gray-100 rounded-md px-2 py-1 inline-block">
                    🎤 {currentAudioDevice}
                  </div>
                  
                  {isScreenRecording && (
                    <div className="text-xs text-gray-600 bg-gray-100 rounded-md px-2 py-1 inline-block">
                      {systemAudioAvailable 
                        ? '🎵 System audio + microphone being captured' 
                        : '🎤 Microphone only (enable "Share system audio" for better results)'}
                    </div>
                  )}
                  
                  {isLiveMode && (
                    <div className="flex items-center justify-center text-xs text-gray-500">
                      <div className={`w-2 h-2 rounded-full mr-1 ${
                        isLiveTranscriptionActive.current ? 'bg-green-500' : 'bg-red-500'
                      }`}></div>
                      Live transcription: {isLiveTranscriptionActive.current ? 'Connected' : 'Disconnected'}
                    </div>
                  )}
                  
                  {isLiveMode && liveSegments.length > 0 && (
                    <div className="text-xs text-blue-600 mt-1">
                      {liveSegments.length} segments transcribed
                    </div>
                  )}
                </div>
              </div>
            )}

            {!isRecording && !isScreenRecording && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mt-4">
                <div className="flex items-start">
                  {Icon('Info', 20, "w-5 h-5 text-blue-500 mr-2 mt-0.5")}
                  <div>
                    <h4 className="text-sm font-medium text-blue-800 mb-1">
                      Tips for Capturing Meeting Audio
                    </h4>
                    <ul className="text-xs text-blue-600 space-y-1">
                      <li>• <strong>Teams/Zoom Meetings:</strong> Use "System Audio Recording" and select "Share system audio"</li>
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

        {((isRecording || isScreenRecording) && isLiveMode) && (
          <Card title="Live Transcript with Timestamps" icon={Icon('Mic')} fullWidth>
            <div className="max-h-80 overflow-y-auto my-2 p-4 bg-gray-50 rounded-lg">
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
                    <span>
                      {isLiveTranscriptionActive.current 
                        ? 'Listening for speech...' 
                        : 'Connecting to live transcription...'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </Card>
        )}
        
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
              {audioUrl && (
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

        {summary && (
          <Card title="Meeting Summary" icon={Icon('FileText')} fullWidth className="mb-4">
            <div className="bg-gray-50 p-4 rounded-lg mb-4">
              <div className="prose max-w-none">
                <ReactMarkdown
                  components={{
                    h1: ({ node, ...props }) => <h1 className="text-xl font-bold mt-4 mb-2" {...props} />,
                    h2: ({ node, ...props }) => <h2 className="text-lg font-bold mt-3 mb-2" {...props} />,
                    h3: ({ node, ...props }) => <h3 className="text-md font-bold mt-2 mb-1" {...props} />,
                    li: ({ node, ...props }) => <li className="ml-4 my-1" {...props} />,
                    p: ({ node, ...props }) => <p className="my-2" {...props} />,
                    strong: ({ node, ...props }) => {
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

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mt-4">
            <div className="flex items-center">
              {Icon('AlertCircle', 20, "w-5 h-5 mr-2")}
              {error}
            </div>
          </div>
        )}

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