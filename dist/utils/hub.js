/**
 * WebInfer.js - Hugging Face Hub Integration
 *
 * Automatically download models, tokenizers, and configs from Hugging Face Hub.
 */
import { loadModelData, isModelCached } from './model-loader.js';
import { Tokenizer } from './tokenizer.js';
import { WebInferError, ErrorCodes } from '../core/types.js';
// ============================================================================
// Constants
// ============================================================================
const DEFAULT_ENDPOINT = 'https://huggingface.co';
const DEFAULT_REVISION = 'main';
/**
 * Common ONNX model file patterns (in order of preference)
 */
const ONNX_MODEL_FILES = [
    'model.onnx',
    'model_quantized.onnx',
    'model_int8.onnx',
    'model_uint8.onnx',
    'model_fp16.onnx',
    'onnx/model.onnx',
    'onnx/model_quantized.onnx',
];
// ============================================================================
// Hub API
// ============================================================================
/**
 * Build URL for a file in a HuggingFace repo
 */
function buildFileUrl(modelId, filename, options = {}) {
    const endpoint = options.endpoint ?? DEFAULT_ENDPOINT;
    const revision = options.revision ?? DEFAULT_REVISION;
    const subfolder = options.subfolder ? `${options.subfolder}/` : '';
    return `${endpoint}/${modelId}/resolve/${revision}/${subfolder}${filename}`;
}
/**
 * Fetch with optional auth token
 */
async function fetchWithAuth(url, token) {
    const headers = {};
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    const response = await fetch(url, { headers });
    return response;
}
/**
 * Check if a file exists in a repo
 */
async function fileExists(modelId, filename, options = {}) {
    const url = buildFileUrl(modelId, filename, options);
    try {
        const response = await fetchWithAuth(url, options.token);
        // HuggingFace returns 302 redirect for existing files
        return response.ok || response.status === 302;
    }
    catch {
        return false;
    }
}
/**
 * Find the best ONNX model file in a repo
 */
async function findOnnxModel(modelId, options = {}) {
    // Try common file patterns
    for (const filename of ONNX_MODEL_FILES) {
        if (await fileExists(modelId, filename, options)) {
            return filename;
        }
    }
    return null;
}
/**
 * Download a file from HuggingFace Hub
 */
export async function downloadFile(modelId, filename, options = {}) {
    const url = buildFileUrl(modelId, filename, options);
    // Use model loader for caching and resume support
    return loadModelData(url, {
        cache: options.cache ?? true,
        forceDownload: options.forceDownload ?? false,
        onProgress: options.onProgress ? (progress) => {
            options.onProgress({
                file: filename,
                fileIndex: 1,
                totalFiles: 1,
                fileProgress: progress,
                overallProgress: progress.percent,
            });
        } : undefined,
    });
}
/**
 * Download JSON file from HuggingFace Hub
 */
export async function downloadJson(modelId, filename, options = {}) {
    const url = buildFileUrl(modelId, filename, options);
    // Check cache first
    if (options.cache !== false && !options.forceDownload) {
        const cached = await isModelCached(url);
        if (cached) {
            const data = await loadModelData(url, { cache: true });
            const text = new TextDecoder().decode(data);
            return JSON.parse(text);
        }
    }
    // Fetch directly for small JSON files
    const response = await fetchWithAuth(url, options.token);
    if (!response.ok) {
        throw new WebInferError(`Failed to download ${filename} from ${modelId}: ${response.status}`, ErrorCodes.MODEL_NOT_FOUND);
    }
    return response.json();
}
/**
 * Download tokenizer from HuggingFace Hub
 */
export async function downloadTokenizer(modelId, options = {}) {
    const url = buildFileUrl(modelId, 'tokenizer.json', options);
    // Use the same cached/resumable loader as models to avoid duplicate downloads
    // across pipelines and sessions.
    const data = await loadModelData(url, {
        cache: options.cache ?? true,
        forceDownload: options.forceDownload ?? false,
        onProgress: options.onProgress ? (progress) => {
            options.onProgress({
                file: 'tokenizer.json',
                fileIndex: 1,
                totalFiles: 1,
                fileProgress: progress,
                overallProgress: progress.percent,
            });
        } : undefined,
    });
    const text = new TextDecoder().decode(data);
    return Tokenizer.fromJSON(text);
}
/**
 * Download model config from HuggingFace Hub
 */
export async function downloadConfig(modelId, options = {}) {
    return downloadJson(modelId, 'config.json', options);
}
/**
 * Download complete model bundle (model + tokenizer + config)
 */
export async function downloadModel(modelId, options = {}) {
    const files = {};
    const totalSteps = 3; // model, tokenizer, config
    let currentStep = 0;
    const reportProgress = (file, progress) => {
        if (options.onProgress) {
            const baseProgress = (currentStep / totalSteps) * 100;
            const stepProgress = (progress.percent / totalSteps);
            options.onProgress({
                file,
                fileIndex: currentStep + 1,
                totalFiles: totalSteps,
                fileProgress: progress,
                overallProgress: baseProgress + stepProgress,
            });
        }
    };
    // 1. Find and download ONNX model
    console.log(`🔍 Finding ONNX model in ${modelId}...`);
    const modelFile = await findOnnxModel(modelId, options);
    if (!modelFile) {
        throw new WebInferError(`No ONNX model found in ${modelId}. Please ensure the model has an ONNX file.`, ErrorCodes.MODEL_NOT_FOUND, { modelId, triedFiles: ONNX_MODEL_FILES });
    }
    files.model = modelFile;
    console.log(`📦 Downloading model: ${modelFile}`);
    const modelData = await downloadFile(modelId, modelFile, {
        ...options,
        onProgress: (p) => reportProgress(modelFile, p.fileProgress),
    });
    currentStep = 1;
    // 2. Download tokenizer (optional)
    let tokenizer;
    try {
        console.log(`📝 Downloading tokenizer...`);
        files.tokenizer = 'tokenizer.json';
        tokenizer = await downloadTokenizer(modelId, options);
        console.log(`✓ Tokenizer loaded`);
    }
    catch (error) {
        console.warn(`⚠️ No tokenizer found for ${modelId}`);
    }
    currentStep = 2;
    // 3. Download config (optional)
    let config;
    try {
        console.log(`⚙️ Downloading config...`);
        files.config = 'config.json';
        config = await downloadConfig(modelId, options);
        console.log(`✓ Config loaded`);
    }
    catch (error) {
        console.warn(`⚠️ No config found for ${modelId}`);
    }
    currentStep = 3;
    if (options.onProgress) {
        options.onProgress({
            file: 'complete',
            fileIndex: totalSteps,
            totalFiles: totalSteps,
            fileProgress: { loaded: 1, total: 1, percent: 100, speed: 0, eta: 0 },
            overallProgress: 100,
        });
    }
    console.log(`✅ Model bundle downloaded: ${modelId}`);
    return {
        modelId,
        modelData,
        tokenizer,
        config,
        files,
    };
}
// ============================================================================
// High-level API
// ============================================================================
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
export async function fromHub(modelId, options = {}) {
    return downloadModel(modelId, options);
}
/**
 * Check if a model exists on HuggingFace Hub
 */
export async function modelExists(modelId, options = {}) {
    try {
        // Try to find an ONNX model
        const modelFile = await findOnnxModel(modelId, options);
        return modelFile !== null;
    }
    catch {
        return false;
    }
}
/**
 * Get model info from HuggingFace Hub
 */
export async function getModelInfo(modelId, options = {}) {
    const [onnxFile, hasTokenizer, config] = await Promise.all([
        findOnnxModel(modelId, options),
        fileExists(modelId, 'tokenizer.json', options),
        downloadConfig(modelId, options).catch(() => undefined),
    ]);
    return {
        hasOnnx: onnxFile !== null,
        onnxFile: onnxFile ?? undefined,
        hasTokenizer,
        hasConfig: config !== undefined,
        config,
    };
}
// ============================================================================
// Popular Models Registry
// ============================================================================
/**
 * Pre-configured popular models
 */
export const POPULAR_MODELS = {
    // Text Classification / Sentiment
    'sentiment-analysis': 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
    'text-classification': 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
    // Feature Extraction
    'feature-extraction': 'Xenova/all-MiniLM-L6-v2',
    'sentence-similarity': 'Xenova/all-MiniLM-L6-v2',
    // Question Answering
    'question-answering': 'Xenova/distilbert-base-cased-distilled-squad',
    // Token Classification
    'ner': 'Xenova/bert-base-NER',
    'token-classification': 'Xenova/bert-base-NER',
    // Text Generation
    'text-generation': 'Xenova/gpt2',
    // Translation
    'translation-en-fr': 'Xenova/t5-small',
    'translation-en-de': 'Xenova/t5-small',
    // Summarization
    'summarization': 'Xenova/distilbart-cnn-6-6',
    // Fill Mask
    'fill-mask': 'Xenova/bert-base-uncased',
    // Image Classification
    'image-classification': 'Xenova/vit-base-patch16-224',
    // Object Detection
    'object-detection': 'Xenova/detr-resnet-50',
    // Image Segmentation
    'image-segmentation': 'Xenova/segformer-b0-finetuned-ade-512-512',
    // Zero-shot Classification
    'zero-shot-classification': 'Xenova/mobilebert-uncased-mnli',
    // Speech Recognition
    'automatic-speech-recognition': 'Xenova/whisper-tiny.en',
    // Text-to-Speech
    'text-to-speech': 'Xenova/speecht5_tts',
};
/**
 * Get the default model ID for a task
 */
export function getDefaultModel(task) {
    return POPULAR_MODELS[task];
}
/**
 * Load a model by task name
 *
 * @example
 * ```typescript
 * const bundle = await fromTask('sentiment-analysis');
 * ```
 */
export async function fromTask(task, options = {}) {
    const modelId = getDefaultModel(task);
    return downloadModel(modelId, options);
}
//# sourceMappingURL=hub.js.map