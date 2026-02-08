/**
 * WebInfer.js - Image Classification Pipeline
 * 
 * Classify images into categories using vision models.
 */

import {
  PipelineConfig,
  PipelineOptions,
} from '../core/types.js';
import { WebInferTensor, softmax } from '../core/tensor.js';
import { ImagePreprocessor, createImagePreprocessor } from '../utils/preprocessor.js';
import {
  BasePipeline,
  ImageClassificationResult,
  registerPipeline,
  IMAGENET_LABELS,
} from './base.js';

// ============================================================================
// Image Classification Pipeline
// ============================================================================

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
export type ImageInput = 
  | HTMLImageElement 
  | HTMLCanvasElement 
  | ImageBitmap 
  | ImageData 
  | string; // URL

/**
 * ImageClassificationPipeline - Classify images
 */
export class ImageClassificationPipeline extends BasePipeline<
  ImageInput | ImageInput[],
  ImageClassificationResult | ImageClassificationResult[]
> {
  private preprocessor: ImagePreprocessor | null = null;
  private labels: string[];
  private numClasses: number;

  constructor(
    config: PipelineConfig, 
    labels?: string[],
    numClasses: number = 1000
  ) {
    super(config);
    this.labels = labels ?? IMAGENET_LABELS;
    this.numClasses = numClasses;
  }

  /**
   * Initialize pipeline
   */
  override async initialize(): Promise<void> {
    await super.initialize();
    
    if (!this.preprocessor) {
      this.preprocessor = createImagePreprocessor('imagenet');
    }
  }

  /**
   * Set custom labels
   */
  setLabels(labels: string[]): void {
    this.labels = labels;
    this.numClasses = labels.length;
  }

  /**
   * Run classification
   */
  override async run(
    input: ImageInput | ImageInput[],
    options?: ImageClassificationOptions
  ): Promise<ImageClassificationResult | ImageClassificationResult[]> {
    const isBatch = Array.isArray(input);
    const inputs = isBatch ? input : [input];
    
    await this.initialize();
    
    const startTime = performance.now();
    const results: ImageClassificationResult[] = [];

    for (const image of inputs) {
      // Preprocess
      const tensorInputs = await this.preprocess(image);
      
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

    return isBatch ? results : results[0]!;
  }

  /**
   * Preprocess image input
   */
  protected override async preprocess(input: ImageInput | ImageInput[]): Promise<WebInferTensor[]> {
    const image = Array.isArray(input) ? input[0]! : input;
    
    // Process image
    const tensor = await this.preprocessor!.process(image);
    
    // Add batch dimension if needed
    if (tensor.shape.length === 3) {
      return [tensor.reshape([1, ...tensor.shape])];
    }
    
    return [tensor];
  }

  /**
   * Run model inference
   */
  private async runInference(inputs: WebInferTensor[]): Promise<WebInferTensor[]> {
    // Generate mock classification logits for demo
    // In production, this would call the actual model
    const logits = new Float32Array(this.numClasses);
    
    // Generate deterministic pseudo-logits based on input
    const inputData = inputs[0]?.toFloat32Array() ?? new Float32Array(0);
    let sum = 0;
    for (let i = 0; i < Math.min(1000, inputData.length); i++) {
      sum += inputData[i] ?? 0;
    }
    
    for (let i = 0; i < this.numClasses; i++) {
      logits[i] = Math.sin(sum * (i + 1) * 0.1) * 3;
    }

    return [new WebInferTensor(logits, [1, this.numClasses], 'float32')];
  }

  /**
   * Postprocess model outputs
   */
  protected override async postprocess(
    outputs: WebInferTensor[],
    options?: ImageClassificationOptions
  ): Promise<ImageClassificationResult> {
    const logits = outputs[0];
    if (!logits) {
      return { label: 'unknown', score: 0 };
    }

    // Apply softmax
    const probs = softmax(logits, -1) as WebInferTensor;
    const probsArray = probs.toFloat32Array();

    const topK = options?.topK ?? 1;

    if (topK > 1 || options?.returnAllScores) {
      // Return top-K results (simplified to top-1 here)
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
// Factory Function
// ============================================================================

/**
 * Create image classification pipeline
 */
export function createImageClassificationPipeline(
  config: Partial<PipelineConfig> = {},
  labels?: string[]
): ImageClassificationPipeline {
  return new ImageClassificationPipeline(
    {
      task: 'image-classification',
      model: config.model ?? 'default',
      runtime: config.runtime,
      cache: config.cache ?? true,
      quantization: config.quantization,
    },
    labels
  );
}

// Register pipeline
registerPipeline('image-classification', (config) => new ImageClassificationPipeline(config));
