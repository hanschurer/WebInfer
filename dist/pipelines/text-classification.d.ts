/**
 * WebInfer.js - Text Classification Pipeline
 *
 * High-level API for text classification tasks including
 * sentiment analysis, topic classification, etc.
 */
import { PipelineConfig, PipelineOptions } from '../core/types.js';
import { WebInferTensor } from '../core/tensor.js';
import { BasePipeline, TextClassificationResult } from './base.js';
/**
 * Text classification options
 */
export interface TextClassificationOptions extends PipelineOptions {
    /** Return all labels with scores */
    returnAllScores?: boolean;
    /** Custom labels */
    labels?: string[];
    /** Number of labels to return */
    topK?: number;
}
/**
 * TextClassificationPipeline - Classify text into categories
 */
export declare class TextClassificationPipeline extends BasePipeline<string | string[], TextClassificationResult | TextClassificationResult[]> {
    private tokenizer;
    private labels;
    constructor(config: PipelineConfig, labels?: string[]);
    /**
     * Initialize pipeline
     */
    initialize(): Promise<void>;
    /**
     * Set custom labels
     */
    setLabels(labels: string[]): void;
    /**
     * Run classification
     */
    run(input: string | string[], options?: TextClassificationOptions): Promise<TextClassificationResult | TextClassificationResult[]>;
    /**
     * Preprocess text input
     */
    protected preprocess(input: string | string[]): Promise<WebInferTensor[]>;
    /**
     * Run model inference
     */
    private runInference;
    /**
     * Postprocess model outputs
     */
    protected postprocess(outputs: WebInferTensor[], options?: TextClassificationOptions): Promise<TextClassificationResult>;
}
/**
 * SentimentAnalysisPipeline - Specialized for sentiment analysis
 */
export declare class SentimentAnalysisPipeline extends TextClassificationPipeline {
    constructor(config: PipelineConfig);
    /**
     * Analyze sentiment
     */
    analyze(text: string | string[], options?: TextClassificationOptions): Promise<TextClassificationResult | TextClassificationResult[]>;
}
/**
 * Create text classification pipeline
 */
export declare function createTextClassificationPipeline(config?: Partial<PipelineConfig>): TextClassificationPipeline;
/**
 * Create sentiment analysis pipeline
 */
export declare function createSentimentAnalysisPipeline(config?: Partial<PipelineConfig>): SentimentAnalysisPipeline;
//# sourceMappingURL=text-classification.d.ts.map