/**
 * WebInfer.js - Base Pipeline
 * 
 * Base class and utilities for all pipeline implementations.
 */

import {
  LoadedModel,
  PipelineConfig,
  PipelineOptions,
  PipelineTask,
} from '../core/types.js';
import { loadModel, runInference } from '../core/runtime.js';
import { WebInferTensor } from '../core/tensor.js';
import { ModelCache } from '../core/memory.js';
import { ModelDownloadCache } from '../utils/cache.js';

// ============================================================================
// Pipeline Types
// ============================================================================

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
  box: { x: number; y: number; width: number; height: number };
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

// ============================================================================
// Base Pipeline Class
// ============================================================================

/**
 * BasePipeline - Abstract base class for all pipelines
 */
export abstract class BasePipeline<TInput, TOutput extends PipelineResult | PipelineResult[]> {
  protected model: LoadedModel | null = null;
  protected readonly config: PipelineConfig;
  protected readonly modelCache: ModelCache;
  protected readonly downloadCache: ModelDownloadCache;
  protected isReady = false;

  constructor(config: PipelineConfig) {
    this.config = config;
    this.modelCache = new ModelCache();
    this.downloadCache = new ModelDownloadCache();
  }

  /**
   * Initialize the pipeline (load model)
   */
  async initialize(): Promise<void> {
    if (this.isReady && this.model) return;

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
  protected async loadModelWithCache(modelPath: string): Promise<LoadedModel> {
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
    } catch {
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
  async run(input: TInput, options?: PipelineOptions): Promise<TOutput> {
    await this.initialize();
    
    const startTime = performance.now();
    
    // Preprocess
    const preprocessed = await this.preprocess(input);
    
    // Run inference
    const outputs = await runInference(this.model!, preprocessed);
    
    // Postprocess
    const result = await this.postprocess(outputs as WebInferTensor[], options);
    
    if (result && typeof result === 'object' && 'processingTime' in result) {
      (result as PipelineResult).processingTime = performance.now() - startTime;
    }
    
    return result;
  }

  /**
   * Run batch inference
   */
  async runBatch(inputs: TInput[], options?: PipelineOptions): Promise<TOutput[]> {
    await this.initialize();
    
    // Process all inputs
    const results = await Promise.all(
      inputs.map(input => this.run(input, options))
    );
    
    return results;
  }

  /**
   * Preprocess input - must be implemented by subclasses
   */
  protected abstract preprocess(input: TInput): Promise<WebInferTensor[]>;

  /**
   * Postprocess output - must be implemented by subclasses
   */
  protected abstract postprocess(
    outputs: WebInferTensor[],
    options?: PipelineOptions
  ): Promise<TOutput>;

  /**
   * Get the task type
   */
  get task(): PipelineTask {
    return this.config.task;
  }

  /**
   * Check if pipeline is ready
   */
  get ready(): boolean {
    return this.isReady;
  }

  /**
   * Dispose the pipeline
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.isReady = false;
  }
}

// ============================================================================
// Pipeline Registry
// ============================================================================

/**
 * Pipeline factory function type
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PipelineFactory = (config: PipelineConfig) => BasePipeline<any, any>;

/**
 * Registered pipeline factories
 */
const pipelineFactories: Map<PipelineTask, PipelineFactory> = new Map();

/**
 * Register a pipeline factory
 */
export function registerPipeline(task: PipelineTask, factory: PipelineFactory): void {
  pipelineFactories.set(task, factory);
}

/**
 * Get a pipeline factory
 */
export function getPipelineFactory(task: PipelineTask): PipelineFactory | undefined {
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
