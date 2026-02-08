/**
 * WebInfer.js - Feature Extraction Pipeline
 *
 * Extract embeddings/features from text, images, or other data.
 */
import { PipelineConfig, PipelineOptions } from '../core/types.js';
import { WebInferTensor } from '../core/tensor.js';
import { BasePipeline, FeatureExtractionResult } from './base.js';
/**
 * Feature extraction options
 */
export interface FeatureExtractionOptions extends PipelineOptions {
    /** Pooling strategy */
    pooling?: 'mean' | 'max' | 'cls' | 'none';
    /** Normalize embeddings */
    normalize?: boolean;
    /** Output dimension (for dimension reduction) */
    outputDim?: number;
}
/**
 * FeatureExtractionPipeline - Extract embeddings from text
 */
export declare class FeatureExtractionPipeline extends BasePipeline<string | string[], FeatureExtractionResult | FeatureExtractionResult[]> {
    private tokenizer;
    private embeddingDim;
    constructor(config: PipelineConfig, embeddingDim?: number);
    /**
     * Initialize pipeline
     */
    initialize(): Promise<void>;
    /**
     * Run feature extraction
     */
    run(input: string | string[], options?: FeatureExtractionOptions): Promise<FeatureExtractionResult | FeatureExtractionResult[]>;
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
    protected postprocess(outputs: WebInferTensor[], options?: FeatureExtractionOptions): Promise<FeatureExtractionResult>;
    /**
     * Extract CLS token embedding
     */
    private extractCLSEmbedding;
    /**
     * Mean pooling over sequence
     */
    private meanPooling;
    /**
     * Max pooling over sequence
     */
    private maxPooling;
    /**
     * L2 normalize vector
     */
    private normalizeVector;
}
/**
 * Create feature extraction pipeline
 */
export declare function createFeatureExtractionPipeline(config?: Partial<PipelineConfig>): FeatureExtractionPipeline;
//# sourceMappingURL=feature-extraction.d.ts.map