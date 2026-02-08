/**
 * WebInfer.js - Text Classification Pipeline
 * 
 * High-level API for text classification tasks including
 * sentiment analysis, topic classification, etc.
 */

import {
  PipelineConfig,
  PipelineOptions,
} from '../core/types.js';
import { WebInferTensor, softmax } from '../core/tensor.js';
import { Tokenizer, createBasicTokenizer } from '../utils/tokenizer.js';
import {
  BasePipeline,
  TextClassificationResult,
  registerPipeline,
  SENTIMENT_LABELS,
} from './base.js';

// ============================================================================
// Text Classification Pipeline
// ============================================================================

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
export class TextClassificationPipeline extends BasePipeline<
  string | string[],
  TextClassificationResult | TextClassificationResult[]
> {
  private tokenizer: Tokenizer | null = null;
  private labels: string[];

  constructor(config: PipelineConfig, labels?: string[]) {
    super(config);
    this.labels = labels ?? SENTIMENT_LABELS;
  }

  /**
   * Initialize pipeline
   */
  override async initialize(): Promise<void> {
    await super.initialize();
    
    // Initialize tokenizer
    if (!this.tokenizer) {
      this.tokenizer = createBasicTokenizer();
    }
  }

  /**
   * Set custom labels
   */
  setLabels(labels: string[]): void {
    this.labels = labels;
  }

  /**
   * Run classification
   */
  override async run(
    input: string | string[],
    options?: TextClassificationOptions
  ): Promise<TextClassificationResult | TextClassificationResult[]> {
    const isBatch = Array.isArray(input);
    const inputs = isBatch ? input : [input];
    
    await this.initialize();
    
    const startTime = performance.now();
    const results: TextClassificationResult[] = [];

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

    return isBatch ? results : results[0]!;
  }

  /**
   * Preprocess text input
   */
  protected override async preprocess(input: string | string[]): Promise<WebInferTensor[]> {
    const text = Array.isArray(input) ? input[0]! : input;
    
    // Tokenize
    const encoded = this.tokenizer!.encode(text, {
      maxLength: 128,
      padding: 'max_length',
      truncation: true,
    });

    // Create tensors
    const inputIds = new WebInferTensor(
      new Float32Array(encoded.inputIds),
      [1, encoded.inputIds.length],
      'float32'
    );

    const attentionMask = new WebInferTensor(
      new Float32Array(encoded.attentionMask),
      [1, encoded.attentionMask.length],
      'float32'
    );

    return [inputIds, attentionMask];
  }

  /**
   * Run model inference
   */
  private async runInference(inputs: WebInferTensor[]): Promise<WebInferTensor[]> {
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
  protected override async postprocess(
    outputs: WebInferTensor[],
    options?: TextClassificationOptions
  ): Promise<TextClassificationResult> {
    const logits = outputs[0];
    if (!logits) {
      return { label: 'unknown', score: 0 };
    }

    // Apply softmax
    const probs = softmax(logits, -1) as WebInferTensor;
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
  constructor(config: PipelineConfig) {
    super(config, SENTIMENT_LABELS);
  }

  /**
   * Analyze sentiment
   */
  async analyze(
    text: string | string[],
    options?: TextClassificationOptions
  ): Promise<TextClassificationResult | TextClassificationResult[]> {
    return this.run(text, options);
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create text classification pipeline
 */
export function createTextClassificationPipeline(
  config: Partial<PipelineConfig> = {}
): TextClassificationPipeline {
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
export function createSentimentAnalysisPipeline(
  config: Partial<PipelineConfig> = {}
): SentimentAnalysisPipeline {
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
