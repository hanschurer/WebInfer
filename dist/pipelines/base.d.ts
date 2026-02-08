/**
 * WebInfer.js - Base Pipeline
 *
 * Base class and utilities for all pipeline implementations.
 */
import { LoadedModel, PipelineConfig, PipelineOptions, PipelineTask } from '../core/types.js';
import { WebInferTensor } from '../core/tensor.js';
import { ModelCache } from '../core/memory.js';
import { ModelDownloadCache } from '../utils/cache.js';
/**
 * Pipeline result base interface
 */
export interface PipelineResult {
    /** Processing time in milliseconds */
    processingTime?: number;
}
/**
 * Text classification result
 */
export interface TextClassificationResult extends PipelineResult {
    label: string;
    score: number;
}
/**
 * Feature extraction result
 */
export interface FeatureExtractionResult extends PipelineResult {
    embeddings: number[];
}
/**
 * Image classification result
 */
export interface ImageClassificationResult extends PipelineResult {
    label: string;
    score: number;
}
/**
 * Object detection result
 */
export interface ObjectDetectionResult extends PipelineResult {
    label: string;
    score: number;
    box: {
        x: number;
        y: number;
        width: number;
        height: number;
    };
}
/**
 * Token classification (NER) span result
 */
export interface TokenClassificationResult extends PipelineResult {
    entity: string;
    score: number;
    start: number;
    end: number;
    word: string;
}
/**
 * BasePipeline - Abstract base class for all pipelines
 */
export declare abstract class BasePipeline<TInput, TOutput extends PipelineResult | PipelineResult[]> {
    protected model: LoadedModel | null;
    protected readonly config: PipelineConfig;
    protected readonly modelCache: ModelCache;
    protected readonly downloadCache: ModelDownloadCache;
    protected isReady: boolean;
    constructor(config: PipelineConfig);
    /**
     * Initialize the pipeline (load model)
     */
    initialize(): Promise<void>;
    /**
     * Load model with caching
     */
    protected loadModelWithCache(modelPath: string): Promise<LoadedModel>;
    /**
     * Run inference (single input)
     */
    run(input: TInput, options?: PipelineOptions): Promise<TOutput>;
    /**
     * Run batch inference
     */
    runBatch(inputs: TInput[], options?: PipelineOptions): Promise<TOutput[]>;
    /**
     * Preprocess input - must be implemented by subclasses
     */
    protected abstract preprocess(input: TInput): Promise<WebInferTensor[]>;
    /**
     * Postprocess output - must be implemented by subclasses
     */
    protected abstract postprocess(outputs: WebInferTensor[], options?: PipelineOptions): Promise<TOutput>;
    /**
     * Get the task type
     */
    get task(): PipelineTask;
    /**
     * Check if pipeline is ready
     */
    get ready(): boolean;
    /**
     * Dispose the pipeline
     */
    dispose(): void;
}
/**
 * Pipeline factory function type
 */
type PipelineFactory = (config: PipelineConfig) => BasePipeline<any, any>;
/**
 * Register a pipeline factory
 */
export declare function registerPipeline(task: PipelineTask, factory: PipelineFactory): void;
/**
 * Get a pipeline factory
 */
export declare function getPipelineFactory(task: PipelineTask): PipelineFactory | undefined;
/**
 * Common sentiment labels
 */
export declare const SENTIMENT_LABELS: string[];
/**
 * Common emotion labels
 */
export declare const EMOTION_LABELS: string[];
/**
 * ImageNet top-10 labels (for demo)
 */
export declare const IMAGENET_LABELS: string[];
export {};
//# sourceMappingURL=base.d.ts.map