import React, { useState, useRef } from 'react';
import { useTranscriber } from "./hooks/useTranscriber";
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
  const [currentStep, setCurrentStep] = useState(0);
  const [summary, setSummary] = useState('');
  const [actionItems, setActionItems] = useState<ActionItem[]>([]);
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
        const parsedItems = parseActionItems(actionItemsText);
        setActionItems(parsedItems);
        
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
    if (transcriber.output?.text) {
      try {
        setError('');
        setIsGeneratingActions(true);
        const items = await generateActionItems(transcriber.output.text);
        setActionItems(items);
        setCurrentStep(5);
      } catch (error) {
        setError('Failed to extract action items. Please try again.');
        console.error('Action items extraction failed:', error);
      } finally {
        setIsGeneratingActions(false);
      }
    }
  };

  // Add type for the line parameter in the action items parsing
  const parseActionItems = (text: string) => {
    // Check if the text is HTML content
    if (text.includes('<div class="action-item">') || text.includes('<p class="task">')) {
      // For HTML content, don't parse and just return the raw HTML
      return text;
    }

    if (!text) return [];

    // Split text into lines and filter out empty lines
    return text.split('\n').filter(line => line.trim().length > 0).map(parseActionItemLine);
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
  
  // Import formatTimestamp from useTranscriber
  import { formatTimestamp } from "./hooks/useTranscriber";

  // Audio player reference
  const audioPlayerRef = useRef<HTMLAudioElement>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  // Function to start audio recording with an optional live transcript parameter
  const startAudioRecording = (withLiveTranscript: boolean) => {
    // First, check if we're already recording
    if (isRecording) {
      console.log('Already recording audio');
      return;
    }
    
    // Set the live mode based on the parameter
    setIsLiveMode(withLiveTranscript);
    
    // Reset transcript and recording state
    transcriber.output = undefined;
    setActionItems([]);
    setSummary('');
    setLiveTranscript('');
    setLiveSegments([]);
    
    console.log(`Starting audio recording. Live transcript: ${withLiveTranscript ? 'ON' : 'OFF'}`);
    
    // Request audio recording permission and set up MediaRecorder
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        // Store the stream in the ref for later use
        const options = { mimeType: 'audio/webm' };
        const mediaRecorder = new MediaRecorder(stream, options);
        mediaRecorderRef.current = mediaRecorder;
        audioChunksRef.current = [];
        
        // If we're in live mode, start the streaming transcription
        if (withLiveTranscript) {
          console.log('Starting streaming transcription with Whisper');
          // Set the engine to 'whisper' for live transcription with timestamps
          transcriber.setEngine('whisper');
          
          transcriber.startStreaming((text, segments) => {
            // This callback will be called with every new transcript update
            setLiveTranscript(text);
            if (segments) {
              setLiveSegments(segments);
            }
          });
        } else {
          // Use faster-whisper for standard transcription
          transcriber.setEngine('faster_whisper');
        }
        
        // Set up event handlers
        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            audioChunksRef.current.push(e.data);
            console.log(`Recorded chunk: ${e.data.size} bytes`);
          }
        };
        
        // Start recording
        mediaRecorder.start(1000); // Collect in 1-second chunks
        setIsRecording(true);
        setCurrentStep(2); // Move to the recording step in the UI
        
        console.log('MediaRecorder started, format:', options.mimeType);
      })
      .catch((error) => {
        console.error('Error accessing microphone:', error);
        alert(`Error accessing microphone: ${error.message}`);
      });
  };

  const stopAudioRecording = async () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      
      // We need to create a new function for ondataavailable that will execute when all data is available
      mediaRecorderRef.current.onstop = () => {
        // Create audio blob and send to transcriber
        if (audioChunksRef.current.length > 0) {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' });
          
          // Create URL for audio playback
          const url = URL.createObjectURL(audioBlob);
          setAudioUrl(url);
          
          // If in live mode, stop streaming and finalize transcript
          if (isLiveMode) {
            transcriber.stopStreaming();
          } else {
            // If not in live mode, transcribe the recorded audio
            transcriber.transcribeFile(audioFile);
          }
        }
        
        setIsRecording(false);
      };
      
      // Stop all tracks
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };

  const startScreenRecording = async (withLiveTranscript: boolean) => {
    try {
      console.log('Starting screen recording with live transcription:', withLiveTranscript);
      
      // Request screen sharing with system audio
      const displayStream = await navigator.mediaDevices.getDisplayMedia({ 
        video: { 
          displaySurface: 'monitor',
          frameRate: { ideal: 30 },
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        },
        audio: true // Try to capture system audio
      });
      
      console.log('Screen stream obtained. Audio tracks:', displayStream.getAudioTracks().length);
      
      // Always capture microphone audio too
      const audioStream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        } 
      });
      
      console.log('Microphone stream obtained. Audio tracks:', audioStream.getAudioTracks().length);
      
      // Combine the streams (both screen video, system audio, and microphone audio)
      const combinedStream = new MediaStream();
      
      // Add all tracks from both streams
      displayStream.getTracks().forEach(track => {
        console.log(`Adding track: ${track.kind} from screen capture`);
        combinedStream.addTrack(track);
      });
      
      audioStream.getTracks().forEach(track => {
        console.log(`Adding track: ${track.kind} from microphone`);
        combinedStream.addTrack(track);
      });
      
      console.log('Combined stream created with tracks:', 
        combinedStream.getVideoTracks().length, 'video,',
        combinedStream.getAudioTracks().length, 'audio');
      
      // Create media recorder with higher bitrate for better quality
      const mediaRecorder = new MediaRecorder(combinedStream, {
        mimeType: 'video/webm;codecs=vp9,opus',
        videoBitsPerSecond: 3000000 // 3 Mbps
      });
      
      screenMediaRecorderRef.current = mediaRecorder;
      screenAudioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          console.log(`Received data chunk: ${e.data.size} bytes`);
          screenAudioChunksRef.current.push(e.data);
        }
      };

      // Start recording - capture data every second
      mediaRecorder.start(1000);
      console.log('Media recorder started');

      if (withLiveTranscript) {
        console.log('Starting live transcription');
        // Start streaming for live transcription with device options
        transcriber.startStreaming(setLiveTranscript, {});
      }
    } catch (error) {
      console.error('Error starting screen recording:', error);
      setIsScreenRecording(false);
    }
  };

  const stopScreenRecording = async () => {
    if (screenMediaRecorderRef.current) {
      screenMediaRecorderRef.current.stop();
      screenMediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      
      // Create video blob and extract audio
      if (screenAudioChunksRef.current.length > 0) {
        const videoBlob = new Blob(screenAudioChunksRef.current, { type: 'video/webm' });
        const videoFile = new File([videoBlob], 'screen-recording.webm', { type: 'video/webm' });
        
        // For screen recording, we need to extract audio and send for transcription
        // In a real app, you'd use a library to extract audio or send the video directly
        // to a backend that can extract the audio
        
        // If in live mode, stop streaming and finalize transcript
        if (isLiveMode) {
          transcriber.stopStreaming();
        } else {
          // In a real implementation, you would extract audio first
          // For now, we'll send the video file directly to demonstrate
          transcriber.transcribeFile(videoFile);
        }
      }

      setIsScreenRecording(false);
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
    const actionItemsText = actionItems.map(item => 
      `Task: ${item.task}\nAssignee: ${item.assignee}\n\n`
    ).join('');
    
    const element = document.createElement("a");
    const file = new Blob([actionItemsText], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = "action_items.txt";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
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
              {liveSegments.length > 0 ? (
                <div className="space-y-2">
                  {liveSegments.map((segment, index) => (
                    <div key={index} className="border-b border-gray-200 pb-2 last:border-0">
                      <div className="text-sm text-gray-500 mb-1">
                        {segment.start !== undefined ? formatTimestamp(segment.start) : '00:00:00'} 
                        {segment.end !== undefined ? ` - ${formatTimestamp(segment.end)}` : ''}
                      </div>
                      <div>{segment.text}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="whitespace-pre-wrap">{liveTranscript}</div>
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
        {((actionItems && actionItems.length > 0) || (typeof actionItems === 'string' && actionItems)) && (
          <Card title="Action Items" icon={Icon('CheckSquare')} fullWidth>
            <div className="space-y-4">
              {Array.isArray(actionItems) ? (
                // Display traditional action items array
                actionItems.map((item, index) => (
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
                ))
              ) : (
                // Display HTML action items
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div dangerouslySetInnerHTML={{ __html: actionItems }}></div>
                </div>
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
