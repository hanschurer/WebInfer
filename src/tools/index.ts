/**
 * WebInfer.js - Tools and Utilities
 * 
 * Model optimization, quantization, and analysis tools.
 */

import { 
  LoadedModel,
  QuantizationType,
} from '../core/types.js';

// ============================================================================
// Quantization Tools
// ============================================================================

/**
 * Quantization options
 */
export interface QuantizationOptions {
  /** Quantization method */
  method: QuantizationType;
  /** Calibration data for calibrated quantization */
  calibrationData?: Float32Array[];
  /** Whether to quantize weights only */
  weightsOnly?: boolean;
  /** Layers to exclude from quantization */
  excludeLayers?: string[];
}

/**
 * Quantization result
 */
export interface QuantizationResult {
  /** Quantized model data */
  modelData: ArrayBuffer;
  /** Original size in bytes */
  originalSize: number;
  /** Quantized size in bytes */
  quantizedSize: number;
  /** Compression ratio */
  compressionRatio: number;
  /** Quantization statistics */
  stats: {
    layersQuantized: number;
    layersSkipped: number;
  };
}

/**
 * Quantize a model
 * 
 * @example
 * ```typescript
 * const quantized = await quantize(model, {
 *   method: 'int8',
 *   calibrationData: samples,
 * });
 * ```
 */
export async function quantize(
  model: LoadedModel | ArrayBuffer,
  options: QuantizationOptions
): Promise<QuantizationResult> {
  // Get model data
  const modelData = model instanceof ArrayBuffer 
    ? model 
    : await getModelData(model);

  const originalSize = modelData.byteLength;

  // Apply quantization based on method
  let quantizedData: ArrayBuffer;
  let layersQuantized = 0;
  let layersSkipped = 0;

  switch (options.method) {
    case 'int8':
      ({ data: quantizedData, layersQuantized, layersSkipped } = 
        quantizeInt8(modelData, options));
      break;
    case 'uint8':
      ({ data: quantizedData, layersQuantized, layersSkipped } = 
        quantizeUint8(modelData, options));
      break;
    case 'float16':
      ({ data: quantizedData, layersQuantized, layersSkipped } = 
        quantizeFloat16(modelData, options));
      break;
    case 'int4':
      ({ data: quantizedData, layersQuantized, layersSkipped } = 
        quantizeInt4(modelData, options));
      break;
    default:
      quantizedData = modelData;
  }

  return {
    modelData: quantizedData,
    originalSize,
    quantizedSize: quantizedData.byteLength,
    compressionRatio: originalSize / quantizedData.byteLength,
    stats: {
      layersQuantized,
      layersSkipped,
    },
  };
}

/**
 * Placeholder for getting model data
 */
async function getModelData(_model: LoadedModel): Promise<ArrayBuffer> {
  // In production, this would extract the model weights
  return new ArrayBuffer(0);
}

/**
 * INT8 quantization
 */
function quantizeInt8(
  data: ArrayBuffer,
  _options: QuantizationOptions
): { data: ArrayBuffer; layersQuantized: number; layersSkipped: number } {
  // Simplified INT8 quantization
  const input = new Float32Array(data);
  const output = new Int8Array(input.length);
  
  // Find scale
  let max = 0;
  for (let i = 0; i < input.length; i++) {
    const abs = Math.abs(input[i] ?? 0);
    if (abs > max) max = abs;
  }
  const scale = max / 127;
  
  // Quantize
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.round((input[i] ?? 0) / scale);
  }
  
  return {
    data: output.buffer,
    layersQuantized: 1,
    layersSkipped: 0,
  };
}

/**
 * UINT8 quantization
 */
function quantizeUint8(
  data: ArrayBuffer,
  _options: QuantizationOptions
): { data: ArrayBuffer; layersQuantized: number; layersSkipped: number } {
  const input = new Float32Array(data);
  const output = new Uint8Array(input.length);
  
  // Find min/max
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < input.length; i++) {
    const val = input[i] ?? 0;
    if (val < min) min = val;
    if (val > max) max = val;
  }
  const scale = (max - min) / 255;
  
  // Quantize
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.round(((input[i] ?? 0) - min) / scale);
  }
  
  return {
    data: output.buffer,
    layersQuantized: 1,
    layersSkipped: 0,
  };
}

/**
 * Float16 quantization
 */
function quantizeFloat16(
  data: ArrayBuffer,
  _options: QuantizationOptions
): { data: ArrayBuffer; layersQuantized: number; layersSkipped: number } {
  const input = new Float32Array(data);
  const output = new Uint16Array(input.length);
  
  // Convert float32 to float16
  for (let i = 0; i < input.length; i++) {
    output[i] = float32ToFloat16(input[i] ?? 0);
  }
  
  return {
    data: output.buffer,
    layersQuantized: 1,
    layersSkipped: 0,
  };
}

/**
 * INT4 quantization
 */
function quantizeInt4(
  data: ArrayBuffer,
  _options: QuantizationOptions
): { data: ArrayBuffer; layersQuantized: number; layersSkipped: number } {
  const input = new Float32Array(data);
  // Pack two INT4 values per byte
  const output = new Uint8Array(Math.ceil(input.length / 2));
  
  // Find scale
  let max = 0;
  for (let i = 0; i < input.length; i++) {
    const abs = Math.abs(input[i] ?? 0);
    if (abs > max) max = abs;
  }
  const scale = max / 7; // INT4 range: -8 to 7
  
  // Quantize and pack
  for (let i = 0; i < input.length; i += 2) {
    const val1 = Math.round((input[i] ?? 0) / scale) + 8;
    const val2 = Math.round((input[i + 1] ?? 0) / scale) + 8;
    output[i / 2] = ((val1 & 0xF) << 4) | (val2 & 0xF);
  }
  
  return {
    data: output.buffer,
    layersQuantized: 1,
    layersSkipped: 0,
  };
}

/**
 * Convert float32 to float16
 */
function float32ToFloat16(value: number): number {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);
  
  floatView[0] = value;
  const x = int32View[0] ?? 0;
  
  let bits = (x >> 16) & 0x8000; // sign
  let m = (x >> 12) & 0x07ff;    // mantissa
  const e = (x >> 23) & 0xff;    // exponent
  
  if (e < 103) {
    // Too small, return zero
    return bits;
  }
  
  if (e > 142) {
    // Too large, return infinity
    bits |= 0x7c00;
    bits |= ((e === 255) ? 0 : 1) && (x & 0x007fffff);
    return bits;
  }
  
  if (e < 113) {
    // Denormalized
    m |= 0x0800;
    bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
    return bits;
  }
  
  bits |= ((e - 112) << 10) | (m >> 1);
  bits += m & 1;
  return bits;
}

// ============================================================================
// Model Pruning
// ============================================================================

/**
 * Pruning options
 */
export interface PruningOptions {
  /** Target sparsity (0-1) */
  sparsity: number;
  /** Pruning method */
  method?: 'magnitude' | 'random' | 'structured';
  /** Layers to exclude */
  excludeLayers?: string[];
}

/**
 * Pruning result
 */
export interface PruningResult {
  /** Pruned model data */
  modelData: ArrayBuffer;
  /** Achieved sparsity */
  actualSparsity: number;
  /** Number of parameters pruned */
  parametersPruned: number;
  /** Total parameters */
  totalParameters: number;
}

/**
 * Prune model weights
 */
export async function prune(
  model: LoadedModel | ArrayBuffer,
  options: PruningOptions
): Promise<PruningResult> {
  const modelData = model instanceof ArrayBuffer 
    ? model 
    : await getModelData(model);

  const weights = new Float32Array(modelData);
  const total = weights.length;
  
  // Calculate threshold for magnitude pruning
  const magnitudes = weights.map(Math.abs);
  const sorted = [...magnitudes].sort((a, b) => a - b);
  const thresholdIdx = Math.floor(options.sparsity * sorted.length);
  const threshold = sorted[thresholdIdx] ?? 0;
  
  // Prune weights
  let pruned = 0;
  for (let i = 0; i < weights.length; i++) {
    if (Math.abs(weights[i] ?? 0) < threshold) {
      weights[i] = 0;
      pruned++;
    }
  }

  return {
    modelData: weights.buffer,
    actualSparsity: pruned / total,
    parametersPruned: pruned,
    totalParameters: total,
  };
}

// ============================================================================
// Model Analysis
// ============================================================================

/**
 * Model analysis result
 */
export interface ModelAnalysis {
  /** Total number of parameters */
  totalParameters: number;
  /** Model size in bytes */
  sizeBytes: number;
  /** Layer information */
  layers: Array<{
    name: string;
    type: string;
    parameters: number;
    inputShape: number[];
    outputShape: number[];
  }>;
  /** Estimated FLOPs */
  estimatedFlops: number;
  /** Memory requirements */
  memoryRequirements: {
    weights: number;
    activations: number;
    total: number;
  };
}

/**
 * Analyze a model
 */
export async function analyzeModel(
  model: LoadedModel | ArrayBuffer
): Promise<ModelAnalysis> {
  // Simplified analysis
  const size = model instanceof ArrayBuffer 
    ? model.byteLength 
    : model.metadata.sizeBytes;

  const estimatedParams = Math.floor(size / 4); // Assume float32

  return {
    totalParameters: estimatedParams,
    sizeBytes: size,
    layers: [],
    estimatedFlops: estimatedParams * 2, // Rough estimate
    memoryRequirements: {
      weights: size,
      activations: size * 0.1, // Rough estimate
      total: size * 1.1,
    },
  };
}

// ============================================================================
// Benchmarking
// ============================================================================

/**
 * Benchmark options
 */
export interface BenchmarkOptions {
  /** Number of warmup runs */
  warmupRuns?: number;
  /** Number of benchmark runs */
  runs?: number;
  /** Input shape */
  inputShape?: number[];
}

/**
 * Benchmark result
 */
export interface BenchmarkResult {
  /** Average inference time in ms */
  avgTime: number;
  /** Minimum inference time in ms */
  minTime: number;
  /** Maximum inference time in ms */
  maxTime: number;
  /** Standard deviation */
  stdDev: number;
  /** Throughput (inferences per second) */
  throughput: number;
  /** All run times */
  times: number[];
}

/**
 * Benchmark model inference
 */
export async function benchmark(
  runFn: () => Promise<void>,
  options: BenchmarkOptions = {}
): Promise<BenchmarkResult> {
  const {
    warmupRuns = 3,
    runs = 10,
  } = options;

  // Warmup
  for (let i = 0; i < warmupRuns; i++) {
    await runFn();
  }

  // Benchmark
  const times: number[] = [];
  for (let i = 0; i < runs; i++) {
    const start = performance.now();
    await runFn();
    times.push(performance.now() - start);
  }

  // Calculate statistics
  const sum = times.reduce((a, b) => a + b, 0);
  const avgTime = sum / times.length;
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times);
  
  const squaredDiffs = times.map(t => Math.pow(t - avgTime, 2));
  const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / times.length;
  const stdDev = Math.sqrt(avgSquaredDiff);

  return {
    avgTime,
    minTime,
    maxTime,
    stdDev,
    throughput: 1000 / avgTime,
    times,
  };
}

// ============================================================================
// Re-export benchmark utilities
// ============================================================================

export {
  benchmark as runBenchmark,
  compareBenchmarks,
  benchmarkSuite,
  benchmarkMemory,
  formatBenchmarkResult,
  formatComparisonResult,
} from './benchmark.js';

export type {
  BenchmarkOptions as DetailedBenchmarkOptions,
  BenchmarkResult as DetailedBenchmarkResult,
  CompareBenchmarkResult,
  MemoryBenchmarkResult,
} from './benchmark.js';

// ============================================================================
// Re-export advanced quantization tools
// ============================================================================

export {
  quantizeModel,
  quantizeTensor,
  dequantizeTensor,
  pruneModel,
  pruneTensor,
  analyzeModel as analyzeModelDetailed,
  exportModel as exportModelAdvanced,
  dequantizeInt8,
  dequantizeUint8,
  dequantizeFloat16,
  float16ToFloat32,
} from './quantization.js';

export type {
  QuantizationType as QuantizationMethod,
  QuantizationOptions as AdvancedQuantizationOptions,
  QuantizationProgress,
  QuantizationResult as AdvancedQuantizationResult,
  LayerQuantizationStats,
  QuantizationStats,
  PruningOptions as AdvancedPruningOptions,
  PruningResult as AdvancedPruningResult,
  ModelAnalysis as DetailedModelAnalysis,
  ExportFormat,
  ExportOptions,
} from './quantization.js';

// ============================================================================
// Re-export debugging tools
// ============================================================================

export {
  WebInferDebugger,
  getDebugger,
  enableDebugging,
  disableDebugging,
  inspectTensor,
  formatTensorInspection,
  createAsciiHistogram,
  createTensorHeatmap,
  visualizeModelArchitecture,
} from './debugger.js';

export type {
  DebuggerConfig,
  TensorInspection,
  TensorStats,
  HistogramData,
  InferenceTrace,
  OperationTrace,
  DebugEvent,
  PerformanceMetrics as DebugPerformanceMetrics,
} from './debugger.js';

// ============================================================================
// Re-export monitoring tools
// ============================================================================

export {
  PerformanceMonitor,
  getMonitor,
  startMonitoring,
  stopMonitoring,
  generateDashboardHTML,
  generateAsciiDashboard,
} from './monitor.js';

export type {
  MonitorConfig,
  PerformanceSample,
  InferenceMetrics,
  MemoryMetrics,
  SystemMetrics,
  AlertConfig,
  AlertEvent,
  WidgetData,
} from './monitor.js';

// ============================================================================
// Export Utilities
// ============================================================================

/**
 * Export model to different formats
 */
export async function exportModel(
  model: LoadedModel | ArrayBuffer,
  format: 'onnx' | 'json' | 'binary'
): Promise<ArrayBuffer | string> {
  const modelData = model instanceof ArrayBuffer 
    ? model 
    : await getModelData(model);

  switch (format) {
    case 'json':
      // Export as JSON (for small models)
      const array = new Float32Array(modelData);
      return JSON.stringify(Array.from(array));
    case 'binary':
    case 'onnx':
    default:
      return modelData;
  }
}
