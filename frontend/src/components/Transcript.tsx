import { useRef, useEffect } from "react";

import { TranscriberData, TranscriptionSegment } from "../hooks/useTranscriber";
import { formatAudioTimestamp } from "../utils/AudioUtils";

interface Props {
    transcriptText: string | undefined;
    segments: TranscriptionSegment[] | undefined;
    // Keeping transcribedData for potential future use with live updates from useTranscriber, but prioritizing direct props for now
    transcribedData?: TranscriberData | undefined; 
}

export default function Transcript({ transcriptText, segments, transcribedData }: Props) {
    const divRef = useRef<HTMLDivElement>(null);

    const saveBlob = (blob: Blob, filename: string) => {
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        link.click();
        URL.revokeObjectURL(url);
    };
    const exportTXT = () => {
        // Prioritize direct segments if available, otherwise fall back to transcribedData.chunks for TXT export
        let exportText = transcriptText || (transcribedData?.chunks?.map(chunk => chunk.text).join('') || '');
        const exportSegments = segments || transcribedData?.chunks?.map(chunk => ({ 
          text: chunk.text, 
          start: chunk.timestamp?.[0],
          end: chunk.timestamp?.[1]
        })) || [];

        let text = exportText
            .trim();

        const blob = new Blob([text], { type: "text/plain" });
        saveBlob(blob, "transcript.txt");
    };
    const exportJSON = () => {
        let jsonData = JSON.stringify(exportSegments, null, 2);

        // post-process the JSON to make it more readable
        const regex = /(    "timestamp": )\[\s+(\S+)\s+(\S+)\s+\]/gm;
        jsonData = jsonData.replace(regex, "$1[$2 $3]");

        const blob = new Blob([jsonData], { type: "application/json" });
        saveBlob(blob, "transcript.json");
    };

    // Scroll to the bottom when the component updates
    useEffect(() => {
        if (divRef.current) {
            const diff = Math.abs(
                divRef.current.offsetHeight +
                    divRef.current.scrollTop -
                    divRef.current.scrollHeight,
            );

            if (diff <= 64) {
                // We're close enough to the bottom, so scroll to the bottom
                divRef.current.scrollTop = divRef.current.scrollHeight;
            }
        }
    });

    // For debugging
    useEffect(() => {
        console.log('Transcript component received props:', {
            text: transcriptText?.substring(0, 50) + '...',
            segmentsCount: segments?.length || 0,
            transcribedDataAvailable: !!transcribedData
        });
    }, [transcriptText, segments, transcribedData]);

    return (
        <div
            ref={divRef}
            className='w-full flex flex-col my-2 p-4 max-h-[20rem] overflow-y-auto'
        >
            {/* Display segments if available */}
            {segments && segments.length > 0 ? (
                segments.map((segment, i) => (
                    <div
                        key={`segment-${i}-${segment.start}`}
                        className='w-full flex flex-row mb-2 bg-white rounded-lg p-4 shadow-xl shadow-black/5 ring-1 ring-slate-700/10'
                    >
                        <div className='mr-5 text-gray-600 tabular-nums'>
                            {formatAudioTimestamp(segment.start || 0)} - {formatAudioTimestamp(segment.end || (segment.start || 0))}
                        </div>
                        {segment.text}
                    </div>
                ))
            ) : transcriptText ? (
                /* Fallback to plain text if no segments but text exists */
                <div className='w-full flex flex-row mb-2 bg-white rounded-lg p-4 shadow-xl shadow-black/5 ring-1 ring-slate-700/10'>
                    <div className='mr-5 text-gray-600 tabular-nums'>
                        {formatAudioTimestamp(0)} 
                    </div>
                    {transcriptText}
                </div>
            ) : null}
            {/* End of segment display logic */}

            {/* No transcript content message */}
            {(!segments || segments.length === 0) && 
             (!transcriptText || transcriptText.trim() === '') && (
                <div className='w-full text-center text-gray-500 italic'>
                    No transcript content available
                </div>
            )}

            {/* Export buttons */}
            {(transcriptText || (segments && segments.length > 0)) && ( // Show export if there's any text or segments
                <div className='w-full text-right mt-4'>
                    <button
                        onClick={exportTXT}
                        className='text-white bg-green-500 hover:bg-green-600 focus:ring-4 focus:ring-green-300 font-medium rounded-lg text-sm px-4 py-2 text-center mr-2 dark:bg-green-600 dark:hover:bg-green-700 dark:focus:ring-green-800 inline-flex items-center'
                    >
                        Export TXT
                    </button>
                    <button
                        onClick={exportJSON}
                        className='text-white bg-green-500 hover:bg-green-600 focus:ring-4 focus:ring-green-300 font-medium rounded-lg text-sm px-4 py-2 text-center mr-2 dark:bg-green-600 dark:hover:bg-green-700 dark:focus:ring-green-800 inline-flex items-center'
                    >
                        Export JSON
                    </button>
                </div>
            )}
        </div>
    );
}
