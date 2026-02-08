/**
 * WebInfer.js - Hugging Face Hub Integration
 *
 * Automatically download models, tokenizers, and configs from Hugging Face Hub.
 */
import { type DownloadProgress } from './model-loader.js';
import { Tokenizer } from './tokenizer.js';
/**
 * Hub options
 */
export interface HubOptions {
    /** HuggingFace API endpoint (default: https://huggingface.co) */
    endpoint?: string;
    /** Model revision/branch (default: main) */
    revision?: string;
    /** Subfolder within the repo */
    subfolder?: string;
    /** Enable caching */
    cache?: boolean;
    /** Force re-download */
    forceDownload?: boolean;
    /** Progress callback */
    onProgress?: (progress: HubDownloadProgress) => void;
    /** HuggingFace API token (for private repos) */
    token?: string;
}
/**
 * Download progress for hub
 */
export interface HubDownloadProgress {
    /** Current file being downloaded */
    file: string;
    /** File index (1-based) */
    fileIndex: number;
    /** Total files */
    totalFiles: number;
    /** File download progress */
    fileProgress: DownloadProgress;
    /** Overall progress (0-100) */
    overallProgress: number;
}
/**
 * Model info from config.json
 */
export interface ModelConfig {
    model_type?: string;
    architectures?: string[];
    hidden_size?: number;
    num_attention_heads?: number;
    num_hidden_layers?: number;
    vocab_size?: number;
    max_position_embeddings?: number;
    type_vocab_size?: number;
    id2label?: Record<string, string>;
    label2id?: Record<string, number>;
    [key: string]: unknown;
}
/**
 * Downloaded model bundle
 */
export interface ModelBundle {
    /** Model ID */
    modelId: string;
    /** Model data (ArrayBuffer) */
    modelData: ArrayBuffer;
    /** Tokenizer instance */
    tokenizer?: Tokenizer;
    /** Model config */
    config?: ModelConfig;
    /** Model files info */
    files: {
        model?: string;
        tokenizer?: string;
        config?: string;
    };
}
/**
 * Download a file from HuggingFace Hub
 */
export declare function downloadFile(modelId: string, filename: string, options?: HubOptions): Promise<ArrayBuffer>;
/**
 * Download JSON file from HuggingFace Hub
 */
export declare function downloadJson<T = unknown>(modelId: string, filename: string, options?: HubOptions): Promise<T>;
/**
 * Download tokenizer from HuggingFace Hub
 */
export declare function downloadTokenizer(modelId: string, options?: HubOptions): Promise<Tokenizer>;
/**
 * Download model config from HuggingFace Hub
 */
export declare function downloadConfig(modelId: string, options?: HubOptions): Promise<ModelConfig>;
/**
 * Download complete model bundle (model + tokenizer + config)
 */
export declare function downloadModel(modelId: string, options?: HubOptions): Promise<ModelBundle>;
/**
 * Load a model from HuggingFace Hub
 *
 * @example
 * ```typescript
 * // Load a sentiment analysis model
 * const bundle = await fromHub('Xenova/distilbert-base-uncased-finetuned-sst-2-english');
 *
 * // Use with WebInfer
 * const model = await loadModelFromBuffer(bundle.modelData);
 * const tokens = bundle.tokenizer.encode('I love this!');
 * ```
 */
export declare function fromHub(modelId: string, options?: HubOptions): Promise<ModelBundle>;
/**
 * Check if a model exists on HuggingFace Hub
 */
export declare function modelExists(modelId: string, options?: HubOptions): Promise<boolean>;
/**
 * Get model info from HuggingFace Hub
 */
export declare function getModelInfo(modelId: string, options?: HubOptions): Promise<{
    hasOnnx: boolean;
    onnxFile?: string;
    hasTokenizer: boolean;
    hasConfig: boolean;
    config?: ModelConfig;
}>;
/**
 * Pre-configured popular models
 */
export declare const POPULAR_MODELS: {
    readonly 'sentiment-analysis': "Xenova/distilbert-base-uncased-finetuned-sst-2-english";
    readonly 'text-classification': "Xenova/distilbert-base-uncased-finetuned-sst-2-english";
    readonly 'feature-extraction': "Xenova/all-MiniLM-L6-v2";
    readonly 'sentence-similarity': "Xenova/all-MiniLM-L6-v2";
    readonly 'question-answering': "Xenova/distilbert-base-cased-distilled-squad";
    readonly ner: "Xenova/bert-base-NER";
    readonly 'token-classification': "Xenova/bert-base-NER";
    readonly 'text-generation': "Xenova/gpt2";
    readonly 'translation-en-fr': "Xenova/t5-small";
    readonly 'translation-en-de': "Xenova/t5-small";
    readonly summarization: "Xenova/distilbart-cnn-6-6";
    readonly 'fill-mask': "Xenova/bert-base-uncased";
    readonly 'image-classification': "Xenova/vit-base-patch16-224";
    readonly 'object-detection': "Xenova/detr-resnet-50";
    readonly 'image-segmentation': "Xenova/segformer-b0-finetuned-ade-512-512";
    readonly 'zero-shot-classification': "Xenova/mobilebert-uncased-mnli";
    readonly 'automatic-speech-recognition': "Xenova/whisper-tiny.en";
    readonly 'text-to-speech': "Xenova/speecht5_tts";
};
export type PopularModelTask = keyof typeof POPULAR_MODELS;
/**
 * Get the default model ID for a task
 */
export declare function getDefaultModel(task: PopularModelTask): string;
/**
 * Load a model by task name
 *
 * @example
 * ```typescript
 * const bundle = await fromTask('sentiment-analysis');
 * ```
 */
export declare function fromTask(task: PopularModelTask, options?: HubOptions): Promise<ModelBundle>;
//# sourceMappingURL=hub.d.ts.map