/**
 * WebInfer.js - Base Pipeline
 *
 * Base class and utilities for all pipeline implementations.
 */
import { loadModel, runInference } from '../core/runtime.js';
import { ModelCache } from '../core/memory.js';
import { ModelDownloadCache } from '../utils/cache.js';
// ============================================================================
// Base Pipeline Class
// ============================================================================
/**
 * BasePipeline - Abstract base class for all pipelines
 */
export class BasePipeline {
    model = null;
    config;
    modelCache;
    downloadCache;
    isReady = false;
    constructor(config) {
        this.config = config;
        this.modelCache = new ModelCache();
        this.downloadCache = new ModelDownloadCache();
    }
    /**
     * Initialize the pipeline (load model)
     */
    async initialize() {
        if (this.isReady && this.model)
            return;
        // Check model cache first
        const cachedModel = this.modelCache.get(this.config.model);
        if (cachedModel) {
            this.model = cachedModel;
            this.isReady = true;
            return;
        }
        // Load model
        this.model = await this.loadModelWithCache(this.config.model);
        this.isReady = true;
    }
    /**
     * Load model with caching
     */
    async loadModelWithCache(modelPath) {
        // Try download cache first
        const cachedResponse = await this.downloadCache.get(modelPath);
        if (cachedResponse) {
            // Use cached data
        }
        // Download and cache (or use mock for now)
        try {
            const response = await fetch(modelPath);
            if (response.ok) {
                // Cache the response
                await this.downloadCache.put(modelPath, response.clone());
            }
        }
        catch {
            // Ignore fetch errors for demo
        }
        // Load into runtime
        return loadModel(modelPath, {
            runtime: this.config.runtime,
            quantization: this.config.quantization,
            cache: this.config.cache,
        });
    }
    /**
     * Run inference (single input)
     */
    async run(input, options) {
        await this.initialize();
        const startTime = performance.now();
        // Preprocess
        const preprocessed = await this.preprocess(input);
        // Run inference
        const outputs = await runInference(this.model, preprocessed);
        // Postprocess
        const result = await this.postprocess(outputs, options);
        if (result && typeof result === 'object' && 'processingTime' in result) {
            result.processingTime = performance.now() - startTime;
        }
        return result;
    }
    /**
     * Run batch inference
     */
    async runBatch(inputs, options) {
        await this.initialize();
        // Process all inputs
        const results = await Promise.all(inputs.map(input => this.run(input, options)));
        return results;
    }
    /**
     * Get the task type
     */
    get task() {
        return this.config.task;
    }
    /**
     * Check if pipeline is ready
     */
    get ready() {
        return this.isReady;
    }
    /**
     * Dispose the pipeline
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        this.isReady = false;
    }
}
/**
 * Registered pipeline factories
 */
const pipelineFactories = new Map();
/**
 * Register a pipeline factory
 */
export function registerPipeline(task, factory) {
    pipelineFactories.set(task, factory);
}
/**
 * Get a pipeline factory
 */
export function getPipelineFactory(task) {
    return pipelineFactories.get(task);
}
// ============================================================================
// Default Label Maps
// ============================================================================
/**
 * Common sentiment labels
 */
export const SENTIMENT_LABELS = ['negative', 'positive'];
/**
 * Common emotion labels
 */
export const EMOTION_LABELS = [
    'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral'
];
/**
 * ImageNet top-10 labels (for demo)
 */
export const IMAGENET_LABELS = [
    'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead',
    'electric ray', 'stingray', 'cock', 'hen', 'ostrich'
];
//# sourceMappingURL=base.js.map