/**
 * WebInfer.js - Image Classification Pipeline
 *
 * Classify images into categories using vision models.
 */
import { PipelineConfig, PipelineOptions } from '../core/types.js';
import { WebInferTensor } from '../core/tensor.js';
import { BasePipeline, ImageClassificationResult } from './base.js';
/**
 * Image classification options
 */
export interface ImageClassificationOptions extends PipelineOptions {
    /** Return all labels with scores */
    returnAllScores?: boolean;
    /** Custom labels */
    labels?: string[];
    /** Number of top predictions to return */
    topK?: number;
}
/**
 * Image classification input types
 */
export type ImageInput = HTMLImageElement | HTMLCanvasElement | ImageBitmap | ImageData | string;
/**
 * ImageClassificationPipeline - Classify images
 */
export declare class ImageClassificationPipeline extends BasePipeline<ImageInput | ImageInput[], ImageClassificationResult | ImageClassificationResult[]> {
    private preprocessor;
    private labels;
    private numClasses;
    constructor(config: PipelineConfig, labels?: string[], numClasses?: number);
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
    run(input: ImageInput | ImageInput[], options?: ImageClassificationOptions): Promise<ImageClassificationResult | ImageClassificationResult[]>;
    /**
     * Preprocess image input
     */
    protected preprocess(input: ImageInput | ImageInput[]): Promise<WebInferTensor[]>;
    /**
     * Run model inference
     */
    private runInference;
    /**
     * Postprocess model outputs
     */
    protected postprocess(outputs: WebInferTensor[], options?: ImageClassificationOptions): Promise<ImageClassificationResult>;
}
/**
 * Create image classification pipeline
 */
export declare function createImageClassificationPipeline(config?: Partial<PipelineConfig>, labels?: string[]): ImageClassificationPipeline;
//# sourceMappingURL=image-classification.d.ts.map