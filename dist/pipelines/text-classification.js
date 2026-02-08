/**
 * WebInfer.js - Text Classification Pipeline
 *
 * High-level API for text classification tasks including
 * sentiment analysis, topic classification, etc.
 */
import { WebInferTensor, softmax } from '../core/tensor.js';
import { createBasicTokenizer } from '../utils/tokenizer.js';
import { BasePipeline, registerPipeline, SENTIMENT_LABELS, } from './base.js';
/**
 * TextClassificationPipeline - Classify text into categories
 */
export class TextClassificationPipeline extends BasePipeline {
    tokenizer = null;
    labels;
    constructor(config, labels) {
        super(config);
        this.labels = labels ?? SENTIMENT_LABELS;
    }
    /**
     * Initialize pipeline
     */
    async initialize() {
        await super.initialize();
        // Initialize tokenizer
        if (!this.tokenizer) {
            this.tokenizer = createBasicTokenizer();
        }
    }
    /**
     * Set custom labels
     */
    setLabels(labels) {
        this.labels = labels;
    }
    /**
     * Run classification
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
        // Add processing time to results
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
        // Tokenize
        const encoded = this.tokenizer.encode(text, {
            maxLength: 128,
            padding: 'max_length',
            truncation: true,
        });
        // Create tensors
        const inputIds = new WebInferTensor(new Float32Array(encoded.inputIds), [1, encoded.inputIds.length], 'float32');
        const attentionMask = new WebInferTensor(new Float32Array(encoded.attentionMask), [1, encoded.attentionMask.length], 'float32');
        return [inputIds, attentionMask];
    }
    /**
     * Run model inference
     */
    async runInference(inputs) {
        // For demo: generate mock logits based on input
        // In production, this would call the actual model
        const numClasses = this.labels.length;
        const logits = new Float32Array(numClasses);
        // Simple sentiment heuristic for demo
        const inputData = inputs[0]?.toFloat32Array() ?? new Float32Array(0);
        const sum = inputData.reduce((a, b) => a + b, 0);
        // Generate pseudo-random but deterministic scores
        for (let i = 0; i < numClasses; i++) {
            logits[i] = Math.sin(sum * (i + 1)) * 2;
        }
        return [new WebInferTensor(logits, [1, numClasses], 'float32')];
    }
    /**
     * Postprocess model outputs
     */
    async postprocess(outputs, options) {
        const logits = outputs[0];
        if (!logits) {
            return { label: 'unknown', score: 0 };
        }
        // Apply softmax
        const probs = softmax(logits, -1);
        const probsArray = probs.toFloat32Array();
        // Get predictions
        const topK = options?.topK ?? 1;
        const returnAllScores = options?.returnAllScores ?? false;
        if (returnAllScores || topK > 1) {
            // Return multiple results - for simplicity, return top-1 here
            // Full implementation would return sorted array
        }
        // Find argmax
        let maxIdx = 0;
        let maxScore = probsArray[0] ?? 0;
        for (let i = 1; i < probsArray.length; i++) {
            if ((probsArray[i] ?? 0) > maxScore) {
                maxScore = probsArray[i] ?? 0;
                maxIdx = i;
            }
        }
        const label = options?.labels?.[maxIdx] ?? this.labels[maxIdx] ?? `class_${maxIdx}`;
        return {
            label,
            score: maxScore,
        };
    }
}
// ============================================================================
// Sentiment Analysis Pipeline
// ============================================================================
/**
 * SentimentAnalysisPipeline - Specialized for sentiment analysis
 */
export class SentimentAnalysisPipeline extends TextClassificationPipeline {
    constructor(config) {
        super(config, SENTIMENT_LABELS);
    }
    /**
     * Analyze sentiment
     */
    async analyze(text, options) {
        return this.run(text, options);
    }
}
// ============================================================================
// Factory Functions
// ============================================================================
/**
 * Create text classification pipeline
 */
export function createTextClassificationPipeline(config = {}) {
    return new TextClassificationPipeline({
        task: 'text-classification',
        model: config.model ?? 'default',
        runtime: config.runtime,
        cache: config.cache ?? true,
        quantization: config.quantization,
    });
}
/**
 * Create sentiment analysis pipeline
 */
export function createSentimentAnalysisPipeline(config = {}) {
    return new SentimentAnalysisPipeline({
        task: 'sentiment-analysis',
        model: config.model ?? 'default',
        runtime: config.runtime,
        cache: config.cache ?? true,
        quantization: config.quantization,
    });
}
// Register pipelines
registerPipeline('text-classification', (config) => new TextClassificationPipeline(config));
registerPipeline('sentiment-analysis', (config) => new SentimentAnalysisPipeline(config));
//# sourceMappingURL=text-classification.js.map