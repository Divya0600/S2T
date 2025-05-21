/* eslint-disable camelcase */
import { pipeline, env } from "@xenova/transformers";

// Disable local models
env.allowLocalModels = false;

// Define model factories
// Ensures only one model is created of each type
class PipelineFactory {
    static task = null;
    static model = null;
    static quantized = null;
    static instance = null;

    constructor(tokenizer, model, quantized) {
        this.tokenizer = tokenizer;
        this.model = model;
        this.quantized = quantized;
    }

    static async getInstance(progress_callback = null) {
        if (this.instance === null) {
            this.instance = pipeline(this.task, this.model, {
                quantized: this.quantized,
                progress_callback,

                // For medium models, we need to load the `no_attentions` revision to avoid running out of memory
                revision: this.model.includes("/whisper-medium") ? "no_attentions" : "main"
            });
        }

        return this.instance;
    }
}

self.addEventListener("message", async (event) => {
    const message = event.data;
    console.log('Worker received message:', {
        modelName: message.model,
        multilingual: message.multilingual,
        quantized: message.quantized,
        subtask: message.subtask,
        language: message.language,
        audioLength: message.audio?.length
    });

    try {
        let transcript = await transcribe(
            message.audio,
            message.model,
            message.multilingual,
            message.quantized,
            message.subtask,
            message.language,
        );
        
        if (transcript === null) {
            console.warn('Transcription returned null');
            return;
        }

        console.log('Transcription completed successfully:', {
            textLength: transcript.text.length,
            chunksCount: transcript.chunks.length
        });

        // Send the result back to the main thread
        self.postMessage({
            status: "complete",
            task: "automatic-speech-recognition",
            data: transcript,
        });
    } catch (error) {
        console.error('Error in worker:', error);
        self.postMessage({
            status: "error",
            task: "automatic-speech-recognition",
            data: error,
        });
    }
});

class AutomaticSpeechRecognitionPipelineFactory extends PipelineFactory {
    static task = "automatic-speech-recognition";
    static model = null;
    static quantized = null;
}

const transcribe = async (
    audio,
    model,
    multilingual,
    quantized,
    subtask,
    language,
) => {
    console.log('Starting transcription with config:', {
        modelType: model,
        multilingual,
        quantized,
        subtask,
        language
    });

    const isDistilWhisper = model.startsWith("distil-whisper/");
    console.log('Using DistilWhisper:', isDistilWhisper);

    let modelName = model;
    if (!isDistilWhisper && !multilingual) {
        modelName += ".en";
        console.log('Using English-only model:', modelName);
    }

    const p = AutomaticSpeechRecognitionPipelineFactory;
    if (p.model !== modelName || p.quantized !== quantized) {
        console.log('Model configuration changed, reinitializing pipeline');
        // Invalidate model if different
        p.model = modelName;
        p.quantized = quantized;

        if (p.instance !== null) {
            console.log('Disposing existing model instance');
            (await p.getInstance()).dispose();
            p.instance = null;
        }
    }

    // Load transcriber model
    console.log('Loading transcriber model');
    let transcriber = await p.getInstance((data) => {
        console.log('Model loading progress:', data);
        self.postMessage(data);
    });
    console.log('Transcriber model loaded successfully');

    const time_precision =
        transcriber.processor.feature_extractor.config.chunk_length /
        transcriber.model.config.max_source_positions;

    // Storage for chunks to be processed. Initialise with an empty chunk.
    let chunks_to_process = [
        {
            tokens: [],
            finalised: false,
        },
    ];

    // TODO: Storage for fully-processed and merged chunks
    // let decoded_chunks = [];

    function chunk_callback(chunk) {
        console.log('Processing chunk:', {
            isLast: chunk.is_last,
            hasTokens: chunk.tokens?.length > 0
        });

        let last = chunks_to_process[chunks_to_process.length - 1];

        // Overwrite last chunk with new info
        Object.assign(last, chunk);
        last.finalised = true;

        // Create an empty chunk after, if it not the last chunk
        if (!chunk.is_last) {
            chunks_to_process.push({
                tokens: [],
                finalised: false,
            });
            console.log('Created new chunk for processing');
        } else {
            console.log('Reached last chunk');
        }
    }

    // Inject custom callback function to handle merging of chunks
    function callback_function(item) {
        let last = chunks_to_process[chunks_to_process.length - 1];

        // Update tokens of last chunk
        last.tokens = [...item[0].output_token_ids];

        // Merge text chunks
        // TODO optimise so we don't have to decode all chunks every time
        let data = transcriber.tokenizer._decode_asr(chunks_to_process, {
            time_precision: time_precision,
            return_timestamps: true,
            force_full_sequences: false,
        });

        self.postMessage({
            status: "update",
            task: "automatic-speech-recognition",
            data: data,
        });
    }

    // Actually run transcription
    console.log('Starting transcription with parameters:', {
        chunkLength: isDistilWhisper ? 20 : 30,
        strideLength: isDistilWhisper ? 3 : 5,
        language,
        task: subtask
    });

    let output = await transcriber(audio, {
        // Greedy
        top_k: 0,
        do_sample: false,

        // Sliding window
        chunk_length_s: isDistilWhisper ? 20 : 30,
        stride_length_s: isDistilWhisper ? 3 : 5,

        // Language and task
        language: language,
        task: subtask,

        // Return timestamps
        return_timestamps: true,
        force_full_sequences: false,

        // Callback functions
        callback_function: callback_function, // after each generation step
        chunk_callback: chunk_callback, // after each chunk is processed
    }).catch((error) => {
        console.error('Transcription error:', error);
        self.postMessage({
            status: "error",
            task: "automatic-speech-recognition",
            data: error,
        });
        return null;
    });

    if (output) {
        console.log('Transcription completed:', {
            textLength: output.text.length,
            chunksCount: output.chunks.length
        });
    }

    return output;
};
