import React, { useState } from 'react';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import useSpeechToTextUX from './useSpeechToTextUX';
import { Mic, Upload, FileText, List, CheckSquare, ChevronRight, BarChart2, Settings, Users, Zap, StopCircle, ArrowLeft, Mail, Save, Edit2 } from 'lucide-react';

const Button = ({ children, onClick, className = '', secondary = false, icon: Icon, as = 'button', ...props }) => {
  const ButtonTag = as;
  return (
    <ButtonTag
      onClick={onClick}
      className={`px-4 py-2 ${secondary ? 'bg-gray-200 text-gray-800' : 'bg-[#5236ab] text-white'} rounded-md shadow-sm hover:shadow-md transition-all duration-200 flex items-center justify-center ${className}`}
      {...props}
    >
      {Icon && <Icon className="mr-2 h-5 w-5" />}
      {children}
    </ButtonTag>
  );
};

const Card = ({ title, children, icon: Icon }) => (
  <div className="bg-white rounded-lg shadow-md p-6 mb-6 border border-[#5236ab]">
    <div className="flex items-center mb-4">
      <Icon className="h-6 w-6 text-[#5236ab] mr-3" />
      <h2 className="text-xl font-semibold text-gray-800">{title}</h2>
    </div>
    {children}
  </div>
);

const ProgressBar = ({ progress }) => (
  <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
    <div
      style={{ width: `${progress}%` }}
      className="bg-[#5236ab] h-2 rounded-full"
    />
  </div>
);

const SpeechToTextUI = () => {
  const {
    step,
    audioFile,
    transcript,
    summary,
    actionItems,
    isProcessing,
    isRecording,
    isLiveStreaming,
    liveTranscript,
    error,
    
    platform,
    jiraUrl,
    jiraUsername,
    uploadedIssues,
    jiraPassword,
    jiraApiToken,
    jiraProjectKey,
    jiraIssueTypes,
    selectedActionItems,
    handlePlatformSelection,
    handleJiraCredentials,
    handleActionItemSelection,
    handleActionItemEdit,
    openAIModel,
    translateOption,
    selectedLanguage,
    emailOption,
    emailContent,
    startRecording,
    stopRecording,
    startLiveStreaming,
    stopLiveStreaming,
    handleFileUpload,
    handleTranslateOption,
    handleLanguageSelection,
    generateTranscript,
    generateAISummary,
    generateAIActionItems,
    uploadToJira,
    handleEmailOption,
    handleEmailSend,
    saveTranscript,
    saveActionItems,
    saveJiraIssues,
    saveSummary,
    goBack,
    resetSession,
    setJiraUrl,
    setStep,
    setJiraUsername,
    setJiraPassword,
   
    setJiraProjectKey,
  } = useSpeechToTextUX();

  const [editingItem, setEditingItem] = useState(null);
  const [emailTo, setEmailTo] = useState('');
  const [emailSubject, setEmailSubject] = useState('Meeting Summary and Action Items');

  const steps = ['Start', 'Audio', 'Translate', 'Transcript', 'Summary', 'Actions', 'Jira', 'Email', 'Complete'];

  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">CGI Data Transmission360</h1>
          <p className="text-lg text-gray-600">Speech to Text powered by AI</p>
        </div>

        <div className="mb-8">
          <ProgressBar progress={(step / (steps.length - 1)) * 100} />
          <div className="flex justify-between">
            {steps.map((s, i) => (
              <div key={s} className={`text-sm ${i <= step ? 'text-[#5236ab] font-medium' : 'text-gray-400'}`}>
                {s}
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4 mb-8">
          <div className="bg-white p-4 rounded-lg shadow-sm flex items-center">
            <BarChart2 className="h-6 w-6 text-[#5236ab] mr-3" />
            <span className="text-sm font-medium text-gray-700">AI Analytics</span>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm flex items-center">
            <Settings className="h-6 w-6 text-[#5236ab] mr-3" />
            <span className="text-sm font-medium text-gray-700">AI Configuration</span>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm flex items-center">
            <Users className="h-6 w-6 text-[#5236ab] mr-3" />
            <span className="text-sm font-medium text-gray-700">AI Team Insights</span>
          </div>
        </div>

        <Card title="Initialize AI Session" icon={Zap}>
          <p className="text-gray-600 mb-4">Choose how you want to input your meeting data:</p>
          <div className="flex justify-center space-x-4">
            {!isRecording && !isLiveStreaming ? (
              <>
                <Button onClick={() => startRecording(true)} icon={Mic}>
                  Start Recording (Display)
                </Button>
                <Button onClick={startLiveStreaming} icon={Mic}>
                  Start Live Streaming
                </Button>
              </>
            ) : (
              <Button 
                onClick={isRecording ? stopRecording : stopLiveStreaming} 
                className="bg-red-500 hover:bg-red-600" 
                icon={StopCircle}
              >
                Stop {isRecording ? 'Recording' : 'Live Streaming'}
              </Button>
            )}
            <Button secondary as="label" htmlFor="file-upload" icon={Upload} className="cursor-pointer">
              Upload Audio
              <input
                id="file-upload"
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={handleFileUpload}
              />
            </Button>
          </div>
        </Card>

        {(isRecording || isLiveStreaming) && (
          <Card title="Live Transcript" icon={FileText}>
            <div className="bg-gray-100 p-4 rounded-lg mb-4 h-64 overflow-y-auto">
            <p className="text-gray-700 whitespace-pre-wrap">{liveTranscript}</p>
            </div>
          </Card>
        )}

        {step >= 1 && (
          <Card title="Audio Source" icon={FileText}>
            <div className="bg-gray-100 p-4 rounded-lg mb-4">
              <span className="font-medium text-gray-700">
                {audioFile ? audioFile.name : 'Recorded Audio'}
              </span>
            </div>
            {audioFile && (
              <audio src={URL.createObjectURL(audioFile)} controls className="w-full mb-4" />
            )}
            {step === 1 && (
              <div className="space-y-4">
                <p className="text-gray-600">Would you like to translate the transcript?</p>
                <div className="flex justify-center space-x-4">
                  <Button onClick={() => handleTranslateOption('Yes')}>Yes</Button>
                  <Button onClick={() => handleTranslateOption('No')} secondary>No</Button>
                </div>
              </div>
            )}
          </Card>
        )}

        {step === 1 && translateOption === 'Yes' && (
          <Card title="Select Language" icon={FileText}>
            <p className="text-gray-600 mb-4">Choose the language for translation:</p>
            <select
              className="w-full p-2 border border-gray-300 rounded-md"
              onChange={(e) => handleLanguageSelection(e.target.value)}
            >
              <option value="">Select a language</option>
              <option value="English">English</option>
              <option value="Spanish">Spanish</option>
              <option value="French">French</option>
              <option value="German">German</option>
              <option value="Chinese">Chinese</option>
              <option value="Japanese">Japanese</option>
              <option value="Korean">Korean</option>
              <option value="Arabic">Arabic</option>
              <option value="Russian">Russian</option>
              <option value="Portuguese">Portuguese</option>
              <option value="Italian">Italian</option>
              <option value="Dutch">Dutch</option>
              <option value="Hindi">Hindi</option>
              <option value="Bengali">Bengali</option>
              <option value="Urdu">Urdu</option>
              <option value="Turkish">Turkish</option>
              <option value="Thai">Thai</option>
              <option value="Vietnamese">Vietnamese</option>
              <option value="Indonesian">Indonesian</option>
            </select>
          </Card>
        )}

        {step >= 2 && (
          <Card title="Transcript" icon={FileText}>
            <div className="bg-gray-100 p-4 rounded-lg mb-4 h-64 overflow-y-auto">
              <p className="text-gray-700">{transcript}</p>
            </div>
            {transcript && (
              <Button onClick={saveTranscript} className="w-full" secondary icon={Save}>
                Save Transcript
              </Button>
            )}
          </Card>
        )}

      
        {step >= 3 && (
          <Card title="AI-Powered Meeting Summary" icon={FileText}>
            <div className="bg-gray-100 p-4 rounded-lg mb-4 h-64 overflow-y-auto">
              <p className="text-gray-700">{summary}</p>
            </div>
            {step === 3 && !summary && (
              <Button onClick={generateAISummary} className="w-full" icon={ChevronRight}>
                Generate AI Summary
              </Button>
            )}
            {summary && (
              <>
                <Button onClick={saveSummary} className="w-full mb-2" secondary icon={Save}>
                  Save Summary
                </Button>
                <Button onClick={() => setStep(4)} className="w-full" icon={ChevronRight}>
                  Next: Extract Action Items
                </Button>
              </>
            )}
          </Card>
        )}

        {step >= 4 && (
          <Card title="AI-Identified Action Items" icon={CheckSquare}>
            <ul className="space-y-2 mb-4">
              {actionItems.map((item, index) => (
                <li key={index} className="flex items-center bg-gray-100 p-3 rounded-lg">
                  <CheckSquare className="h-5 w-5 text-[#5236ab] mr-3" />
                  <span className="text-gray-700">
                    {item.task} - <span className="font-medium text-gray-900">{item.assignee}</span>
                  </span>
                </li>
              ))}
            </ul>
            {step === 4 && !actionItems.length && (
              <Button onClick={generateAIActionItems} className="w-full mb-2" icon={ChevronRight}>
                Extract AI Action Items
              </Button>
            )}
            {actionItems.length > 0 && (
              <>
                <Button onClick={saveActionItems} className="w-full mb-2" secondary icon={Save}>
                  Save Action Items
                </Button>
                <Button onClick={() => setStep(5)} className="w-full" icon={ChevronRight}>
                  Next: Upload to Jira
                </Button>
              </>
            )}
          </Card>
        )}
        {step === 5 && (
          <Card title="Upload Action Items" icon={Upload}>
            <p className="text-gray-600 mb-4">Do you want to upload the action items as issues?</p>
            <div className="flex justify-center space-x-4">
              <Button onClick={() => handlePlatformSelection('Jira')}>Jira</Button>
              <Button onClick={() => handlePlatformSelection('Azure DevOps')}>Azure DevOps</Button>
              <Button onClick={() => setStep(6)} secondary>Skip</Button>
            </div>
          </Card>
        )}

        {step === 6 && (
          <Card title="Connect to Jira" icon={Settings}>
            <input
              type="text"
              placeholder="Jira Project Key"
              value={jiraProjectKey}
              onChange={(e) => setJiraProjectKey(e.target.value)}
              className="w-full p-2 mb-4 border rounded"
            />
            <Button onClick={() => handleJiraCredentials(jiraProjectKey)} className="w-full">
              Connect to Jira
            </Button>
          </Card>
        )}

        {step === 7 && (
          <Card title="Select Action Items to Upload" icon={CheckSquare}>
            {actionItems.map((item, index) => (
              <div key={index} className="flex items-center mb-2">
                <input
                  type="checkbox"
                  checked={selectedActionItems[index] || false}
                  onChange={(e) => handleActionItemSelection(index, e.target.checked)}
                  className="mr-2"
                />
                {editingItem === index ? (
                  <div className="flex-grow">
                    <input
                      type="text"
                      value={item.task}
                      onChange={(e) => handleActionItemEdit(index, e.target.value, item.assignee, item.issueType)}
                      className="w-full p-1 border rounded mb-1"
                    />
                    <input
                      type="email"
                      value={item.assignee}
                      onChange={(e) => handleActionItemEdit(index, item.task, e.target.value, item.issueType)}
                      placeholder="Assignee email"
                      className="w-full p-1 border rounded mb-1"
                    />
                    <select
                      value={item.issueType || ''}
                      onChange={(e) => handleActionItemEdit(index, item.task, item.assignee, e.target.value)}
                      className="w-full p-1 border rounded mb-1"
                    >
                      <option value="">Select Issue Type</option>
                      {jiraIssueTypes.map((type) => (
                        <option key={type.id} value={type.name}>{type.name}</option>
                      ))}
                    </select>
                    <Button onClick={() => setEditingItem(null)} className="mt-1">Save</Button>
                  </div>
                ) : (
                  <>
                    <span className="flex-grow">{item.task} - {item.assignee} ({item.issueType || 'Task'})</span>
                    <Button onClick={() => setEditingItem(index)} secondary icon={Edit2}>Edit</Button>
                  </>
                )}
              </div>
            ))}
            <Button onClick={uploadToJira} className="w-full mt-4">Upload Selected to Jira</Button>
          </Card>
        )}

        {step === 8 && (
          <Card title="Upload Complete" icon={CheckSquare}>
            <div className="text-center mb-4">
              <CheckSquare className="h-16 w-16 text-green-500 mx-auto mb-4" />
              <p className="text-xl font-medium text-gray-800 mb-2">
                All tasks have been successfully uploaded to Jira!
              </p>
            </div>
            
            <h3 className="text-lg font-semibold mb-2">Uploaded Issues:</h3>
            <ul className="space-y-2 mb-4">
              {uploadedIssues.map((issue, index) => (
                <li key={index} className="bg-white p-3 rounded-lg shadow">
                  <p className="font-medium">{issue.task}</p>
                  <p className="text-sm text-gray-600">Assignee: {issue.assignee}</p>
                  <p className="text-sm text-gray-600">Issue Type: {issue.issueType || 'Task'}</p>
                  <p className="text-sm font-semibold text-blue-600">Jira Key: {issue.issueKey}</p>
                </li>
              ))}
            </ul>
            
            <div className="space-y-2">
              <Button onClick={saveJiraIssues} className="w-full" secondary icon={Save}>
                Save Jira Issues
              </Button>
              <Button onClick={resetSession} className="w-full" icon={Zap}>
                Start New Session
              </Button>
            </div>
          </Card>
        )}
            
          

        {step > 0 && step < 8 && (
          <Button onClick={goBack} secondary className="mt-4" icon={ArrowLeft}>
            Go Back
          </Button>
        )}

        {openAIModel && (
          <div className="text-sm text-gray-600 mt-2">
            OpenAI Model Used: {openAIModel}
          </div>
        )}

        {isProcessing && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
            <div className="bg-white rounded-lg p-8 flex flex-col items-center">
              <div className="w-12 h-12 border-t-4 border-[#5236ab] border-solid rounded-full animate-spin mb-4"></div>
              <p className="text-gray-800 text-lg">AI Processing...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mt-4" role="alert">
            <strong className="font-bold">Error: </strong>
            <span className="block sm:inline">{error}</span>
          </div>
        )}

        <ToastContainer position="top-right" autoClose={1000} />
      </div>
    </div>
  );
};

export default SpeechToTextUI;
