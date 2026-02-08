/**
 * WebInfer.js - Core Type Definitions
 * 
 * This file contains all the core types used throughout the framework.
 */

// ============================================================================
// Tensor Types
// ============================================================================

/**
 * Supported data types for tensors
 */
export type DataType = 
  | 'float32' 
  | 'float16' 
  | 'int32' 
  | 'int64' 
  | 'uint8' 
  | 'int8' 
  | 'bool';

/**
 * TypedArray types used for tensor data
 */
export type TypedArray = 
  | Float32Array 
  | Float64Array 
  | Int32Array 
  | BigInt64Array 
  | Uint8Array 
  | Int8Array;

/**
 * Tensor shape definition
 */
export type Shape = readonly number[];

/**
 * Tensor interface
 */
export interface Tensor {
  /** Unique identifier for the tensor */
  readonly id: string;
  /** Data type of the tensor */
  readonly dtype: DataType;
  /** Shape of the tensor */
  readonly shape: Shape;
  /** Total number of elements */
  readonly size: number;
  /** Underlying data */
  readonly data: TypedArray;
  /** Get data as Float32Array */
  toFloat32Array(): Float32Array;
  /** Get data as array */
  toArray(): number[];
  /** Clone the tensor */
  clone(): Tensor;
  /** Dispose the tensor and free memory */
  dispose(): void;
  /** Check if tensor has been disposed */
  readonly isDisposed: boolean;
}

// ============================================================================
// Runtime Types
// ============================================================================

/**
 * Supported runtime backends
 */
export type RuntimeType = 'webgpu' | 'webnn' | 'wasm' | 'auto';

/**
 * Runtime capability flags
 */
export interface RuntimeCapabilities {
  /** Supports concurrent execution */
  concurrency: boolean;
  /** Supports quantized models */
  quantization: boolean;
  /** Supports float16 */
  float16: boolean;
  /** Supports dynamic shapes */
  dynamicShapes: boolean;
  /** Maximum batch size */
  maxBatchSize: number;
  /** Available memory in bytes */
  availableMemory: number;
}

/**
 * Runtime interface that all backends must implement
 */
export interface Runtime {
  /** Runtime name */
  readonly name: RuntimeType;
  /** Runtime capabilities */
  readonly capabilities: RuntimeCapabilities;
  /** Initialize the runtime */
  initialize(): Promise<void>;
  /** Check if runtime is available in current environment */
  isAvailable(): Promise<boolean>;
  /** Load a model from ArrayBuffer */
  loadModel(modelData: ArrayBuffer, options?: ModelLoadOptions): Promise<LoadedModel>;
  /** Run inference */
  run(model: LoadedModel, inputs: Tensor[]): Promise<Tensor[]>;
  /** Dispose the runtime and free resources */
  dispose(): void;
}

// ============================================================================
// Model Types
// ============================================================================

/**
 * Model format types
 */
export type ModelFormat = 'onnx' | 'WebInfer' | 'safetensors';

/**
 * Model quantization types
 */
export type QuantizationType = 'float32' | 'float16' | 'int8' | 'uint8' | 'int4';

/**
 * Model metadata
 */
export interface ModelMetadata {
  /** Model name/identifier */
  name: string;
  /** Model version */
  version?: string;
  /** Model description */
  description?: string;
  /** Model author */
  author?: string;
  /** Model license */
  license?: string;
  /** Model tags */
  tags?: string[];
  /** Input specifications */
  inputs: ModelIOSpec[];
  /** Output specifications */
  outputs: ModelIOSpec[];
  /** Model size in bytes */
  sizeBytes: number;
  /** Quantization type */
  quantization: QuantizationType;
  /** Model format */
  format: ModelFormat;
}

/**
 * Model input/output specification
 */
export interface ModelIOSpec {
  /** Name of the input/output */
  name: string;
  /** Data type */
  dtype: DataType;
  /** Shape (use -1 for dynamic dimensions) */
  shape: number[];
  /** Optional description */
  description?: string;
}

/**
 * Options for loading a model
 */
export interface ModelLoadOptions {
  /** Target quantization (convert during load) */
  quantization?: QuantizationType;
  /** Custom metadata */
  metadata?: Partial<ModelMetadata>;
  /** Enable caching */
  cache?: boolean;
  /** Progress callback */
  onProgress?: (progress: number) => void;
}

/**
 * Loaded model instance
 */
export interface LoadedModel {
  /** Unique model instance ID */
  readonly id: string;
  /** Model metadata */
  readonly metadata: ModelMetadata;
  /** Check if model is loaded */
  readonly isLoaded: boolean;
  /** Runtime this model is loaded on */
  readonly runtime: RuntimeType;
  /** Dispose the model and free resources */
  dispose(): void;
}

// ============================================================================
// Scheduler Types
// ============================================================================

/**
 * Task priority levels
 */
export type TaskPriority = 'low' | 'normal' | 'high' | 'critical';

/**
 * Task status
 */
export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

/**
 * Inference task definition
 */
export interface InferenceTask<T = unknown> {
  /** Unique task ID */
  readonly id: string;
  /** Model ID this task is for */
  readonly modelId: string;
  /** Task priority */
  readonly priority: TaskPriority;
  /** Task status */
  readonly status: TaskStatus;
  /** Creation timestamp */
  readonly createdAt: number;
  /** Start timestamp (when running) */
  readonly startedAt?: number;
  /** Completion timestamp */
  readonly completedAt?: number;
  /** Task result (when completed) */
  readonly result?: T;
  /** Task error (when failed) */
  readonly error?: Error;
  /** Cancel the task */
  cancel(): void;
  /** Wait for task completion */
  wait(): Promise<T>;
}

/**
 * Scheduler options
 */
export interface SchedulerOptions {
  /** Maximum concurrent tasks across all models */
  maxConcurrentTasks?: number;
  /** Maximum concurrent tasks per model */
  maxConcurrentPerModel?: number;
  /** Default task timeout in milliseconds */
  defaultTimeout?: number;
  /** Enable task batching */
  enableBatching?: boolean;
  /** Maximum batch size */
  maxBatchSize?: number;
  /** Batch timeout in milliseconds */
  batchTimeout?: number;
}

// ============================================================================
// Memory Types
// ============================================================================

/**
 * Memory statistics
 */
export interface MemoryStats {
  /** Total allocated memory in bytes */
  allocated: number;
  /** Currently used memory in bytes */
  used: number;
  /** Peak memory usage in bytes */
  peak: number;
  /** Number of active tensors */
  tensorCount: number;
  /** Number of loaded models */
  modelCount: number;
}

/**
 * Memory pool configuration
 */
export interface MemoryPoolConfig {
  /** Initial pool size in bytes */
  initialSize?: number;
  /** Maximum pool size in bytes */
  maxSize?: number;
  /** Growth factor when expanding */
  growthFactor?: number;
  /** Enable automatic garbage collection */
  autoGC?: boolean;
  /** GC threshold (percentage of max size) */
  gcThreshold?: number;
}

// ============================================================================
// Pipeline Types
// ============================================================================

/**
 * Supported pipeline tasks
 */
export type PipelineTask = 
  | 'text-classification'
  | 'token-classification'
  | 'question-answering'
  | 'fill-mask'
  | 'text-generation'
  | 'text2text-generation'
  | 'summarization'
  | 'translation'
  | 'feature-extraction'
  | 'sentiment-analysis'
  | 'zero-shot-classification'
  | 'image-classification'
  | 'object-detection'
  | 'image-segmentation'
  | 'depth-estimation'
  | 'image-to-text'
  | 'audio-classification'
  | 'automatic-speech-recognition'
  | 'text-to-speech';

/**
 * Pipeline configuration
 */
export interface PipelineConfig {
  /** Task type */
  task: PipelineTask;
  /** Model ID or path */
  model: string;
  /** Runtime to use */
  runtime?: RuntimeType;
  /** Enable caching */
  cache?: boolean;
  /** Quantization type */
  quantization?: QuantizationType;
  /** Device to use */
  device?: 'cpu' | 'gpu';
  /** Custom tokenizer config */
  tokenizer?: TokenizerConfig;
}

/**
 * Pipeline options passed during inference
 */
export interface PipelineOptions {
  /** Batch size */
  batchSize?: number;
  /** Top K results */
  topK?: number;
  /** Temperature for generation */
  temperature?: number;
  /** Maximum length for generation */
  maxLength?: number;
  /** Task timeout in milliseconds */
  timeout?: number;
}

// ============================================================================
// Tokenizer Types
// ============================================================================

/**
 * Tokenizer configuration
 */
export interface TokenizerConfig {
  /** Vocabulary size */
  vocabSize: number;
  /** Maximum sequence length */
  maxLength: number;
  /** Padding token ID */
  padTokenId: number;
  /** Unknown token ID */
  unkTokenId: number;
  /** Start of sequence token ID */
  bosTokenId?: number;
  /** End of sequence token ID */
  eosTokenId?: number;
  /** Separator token ID */
  sepTokenId?: number;
  /** CLS token ID */
  clsTokenId?: number;
  /** Mask token ID */
  maskTokenId?: number;
}

/**
 * Tokenized output
 */
export interface TokenizedOutput {
  /** Input IDs */
  inputIds: number[];
  /** Attention mask */
  attentionMask: number[];
  /** Token type IDs (for segment embeddings) */
  tokenTypeIds?: number[];
  /** Special tokens mask */
  specialTokensMask?: number[];
  /** Offset mapping (for token-level tasks) */
  offsetMapping?: [number, number][];
}

// ============================================================================
// Error Types
// ============================================================================

/**
 * Base error class for WebInfer errors
 */
export class WebInferError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'WebInferError';
  }
}

/**
 * Error codes
 */
export const ErrorCodes = {
  // Runtime errors
  RUNTIME_NOT_AVAILABLE: 'RUNTIME_NOT_AVAILABLE',
  RUNTIME_INIT_FAILED: 'RUNTIME_INIT_FAILED',
  RUNTIME_NOT_INITIALIZED: 'RUNTIME_NOT_INITIALIZED',
  
  // Model errors
  MODEL_NOT_FOUND: 'MODEL_NOT_FOUND',
  MODEL_LOAD_FAILED: 'MODEL_LOAD_FAILED',
  MODEL_INVALID_FORMAT: 'MODEL_INVALID_FORMAT',
  MODEL_NOT_LOADED: 'MODEL_NOT_LOADED',
  
  // Inference errors
  INFERENCE_FAILED: 'INFERENCE_FAILED',
  INFERENCE_TIMEOUT: 'INFERENCE_TIMEOUT',
  INFERENCE_CANCELLED: 'INFERENCE_CANCELLED',
  
  // Memory errors
  OUT_OF_MEMORY: 'OUT_OF_MEMORY',
  MEMORY_LEAK_DETECTED: 'MEMORY_LEAK_DETECTED',
  
  // Tensor errors
  TENSOR_SHAPE_MISMATCH: 'TENSOR_SHAPE_MISMATCH',
  TENSOR_DTYPE_MISMATCH: 'TENSOR_DTYPE_MISMATCH',
  TENSOR_DISPOSED: 'TENSOR_DISPOSED',
  
  // Pipeline errors
  PIPELINE_NOT_SUPPORTED: 'PIPELINE_NOT_SUPPORTED',
  PIPELINE_INPUT_INVALID: 'PIPELINE_INPUT_INVALID',
  
  // General errors
  INVALID_ARGUMENT: 'INVALID_ARGUMENT',
  NOT_IMPLEMENTED: 'NOT_IMPLEMENTED',
  UNKNOWN_ERROR: 'UNKNOWN_ERROR',
} as const;

export type ErrorCode = typeof ErrorCodes[keyof typeof ErrorCodes];

// ============================================================================
// Event Types
// ============================================================================

/**
 * Event types emitted by WebInfer
 */
export type EventType = 
  | 'model:loading'
  | 'model:loaded'
  | 'model:unloaded'
  | 'inference:start'
  | 'inference:complete'
  | 'inference:error'
  | 'memory:warning'
  | 'memory:gc'
  | 'runtime:ready'
  | 'runtime:error';

/**
 * Event payload interface
 */
export interface WebInferEvent<T = unknown> {
  type: EventType;
  timestamp: number;
  data: T;
}

/**
 * Event listener function type
 */
export type EventListener<T = unknown> = (event: WebInferEvent<T>) => void;
