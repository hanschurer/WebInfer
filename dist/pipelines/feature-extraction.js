/**
 * WebInfer.js - Feature Extraction Pipeline
 *
 * Extract embeddings/features from text, images, or other data.
 */
import { WebInferTensor } from '../core/tensor.js';
import { createBasicTokenizer } from '../utils/tokenizer.js';
import { BasePipeline, registerPipeline, } from './base.js';
/**
 * FeatureExtractionPipeline - Extract embeddings from text
 */
export class FeatureExtractionPipeline extends BasePipeline {
    tokenizer = null;
    embeddingDim;
    constructor(config, embeddingDim = 768) {
        super(config);
        this.embeddingDim = embeddingDim;
    }
    /**
     * Initialize pipeline
     */
    async initialize() {
        await super.initialize();
        if (!this.tokenizer) {
            this.tokenizer = createBasicTokenizer();
        }
    }
    /**
     * Run feature extraction
     */
    async run(input, options) {
        const isBatch = Array.isArray(input);
        const inputs = isBatch ? input : [input];
        await this.initialize();
        const startTime = performance.now();
        const results = [];
        for (const text of inputs) {
            // Preprocess
            const tensorInputs = await this.preprocess(text);
            // Run inference
            const outputs = await this.runInference(tensorInputs);
            // Postprocess
            const result = await this.postprocess(outputs, options);
            results.push(result);
        }
        const processingTime = performance.now() - startTime;
        for (const result of results) {
            result.processingTime = processingTime / results.length;
        }
        return isBatch ? results : results[0];
    }
    /**
     * Preprocess text input
     */
    async preprocess(input) {
        const text = Array.isArray(input) ? input[0] : input;
        const encoded = this.tokenizer.encode(text, {
            maxLength: 128,
            padding: 'max_length',
            truncation: true,
        });
        const inputIds = new WebInferTensor(new Float32Array(encoded.inputIds), [1, encoded.inputIds.length], 'float32');
        const attentionMask = new WebInferTensor(new Float32Array(encoded.attentionMask), [1, encoded.attentionMask.length], 'float32');
        return [inputIds, attentionMask];
    }
    /**
     * Run model inference
     */
    async runInference(inputs) {
        // Generate mock embeddings for demo
        // In production, this would call the actual model
        const seqLen = inputs[0]?.shape[1] ?? 128;
        const embeddings = new Float32Array(seqLen * this.embeddingDim);
        // Generate deterministic pseudo-embeddings based on input
        const inputData = inputs[0]?.toFloat32Array() ?? new Float32Array(0);
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < this.embeddingDim; j++) {
                const inputVal = inputData[i] ?? 0;
                embeddings[i * this.embeddingDim + j] =
                    Math.sin(inputVal * (j + 1) * 0.01) * 0.1;
            }
        }
        return [new WebInferTensor(embeddings, [1, seqLen, this.embeddingDim], 'float32')];
    }
    /**
     * Postprocess model outputs
     */
    async postprocess(outputs, options) {
        const hiddenStates = outputs[0];
        if (!hiddenStates) {
            return { embeddings: [] };
        }
        const pooling = options?.pooling ?? 'mean';
        const normalize = options?.normalize ?? true;
        let embeddings;
        switch (pooling) {
            case 'cls':
                // Use first token (CLS) embedding
                embeddings = this.extractCLSEmbedding(hiddenStates);
                break;
            case 'max':
                // Max pooling
                embeddings = this.maxPooling(hiddenStates);
                break;
            case 'none':
                // Return all token embeddings (flattened)
                embeddings = hiddenStates.toArray();
                break;
            case 'mean':
            default:
                // Mean pooling
                embeddings = this.meanPooling(hiddenStates);
                break;
        }
        // Normalize if requested
        if (normalize) {
            embeddings = this.normalizeVector(embeddings);
        }
        // Dimension reduction if requested
        if (options?.outputDim && options.outputDim < embeddings.length) {
            embeddings = embeddings.slice(0, options.outputDim);
        }
        return { embeddings };
    }
    /**
     * Extract CLS token embedding
     */
    extractCLSEmbedding(hiddenStates) {
        const data = hiddenStates.toFloat32Array();
        const embeddingDim = hiddenStates.shape[2] ?? this.embeddingDim;
        return Array.from(data.slice(0, embeddingDim));
    }
    /**
     * Mean pooling over sequence
     */
    meanPooling(hiddenStates) {
        const data = hiddenStates.toFloat32Array();
        const seqLen = hiddenStates.shape[1] ?? 1;
        const embeddingDim = hiddenStates.shape[2] ?? this.embeddingDim;
        const result = new Float32Array(embeddingDim);
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < embeddingDim; j++) {
                result[j] = (result[j] ?? 0) + (data[i * embeddingDim + j] ?? 0) / seqLen;
            }
        }
        return Array.from(result);
    }
    /**
     * Max pooling over sequence
     */
    maxPooling(hiddenStates) {
        const data = hiddenStates.toFloat32Array();
        const seqLen = hiddenStates.shape[1] ?? 1;
        const embeddingDim = hiddenStates.shape[2] ?? this.embeddingDim;
        const result = new Array(embeddingDim).fill(-Infinity);
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < embeddingDim; j++) {
                const val = data[i * embeddingDim + j] ?? 0;
                if (val > (result[j] ?? -Infinity)) {
                    result[j] = val;
                }
            }
        }
        return result;
    }
    /**
     * L2 normalize vector
     */
    normalizeVector(vec) {
        let norm = 0;
        for (const v of vec) {
            norm += v * v;
        }
        norm = Math.sqrt(norm);
        if (norm === 0)
            return vec;
        return vec.map(v => v / norm);
    }
}
// ============================================================================
// Factory Function
// ============================================================================
/**
 * Create feature extraction pipeline
 */
export function createFeatureExtractionPipeline(config = {}) {
    return new FeatureExtractionPipeline({
        task: 'feature-extraction',
        model: config.model ?? 'default',
        runtime: config.runtime,
        cache: config.cache ?? true,
        quantization: config.quantization,
    });
}
// Register pipeline
registerPipeline('feature-extraction', (config) => new FeatureExtractionPipeline(config));
//# sourceMappingURL=feature-extraction.js.map