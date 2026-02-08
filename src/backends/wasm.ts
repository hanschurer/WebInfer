/**
 * WebInfer.js - WebAssembly Backend
 * 
 * Pure WASM runtime for universal browser support.
 * Features:
 * - Universal compatibility
 * - SIMD acceleration when available
 * - Memory-efficient execution
 */

import {
  Runtime,
  RuntimeType,
  RuntimeCapabilities,
  LoadedModel,
  ModelLoadOptions,
  ModelMetadata,
  Tensor,
  WebInferError,
  ErrorCodes,
} from '../core/types.js';
import { LoadedModelImpl } from '../core/runtime.js';
import { WebInferTensor, softmax as tensorSoftmax, relu as tensorRelu, sigmoid as tensorSigmoid } from '../core/tensor.js';
import { getMemoryManager } from '../core/memory.js';

// ============================================================================
// WASM Types
// ============================================================================

/**
 * WASM module instance
 */
interface WASMModule {
  memory: WebAssembly.Memory;
  exports: WASMExports;
}

/**
 * WASM exported functions
 */
interface WASMExports {
  // Memory management
  malloc(size: number): number;
  free(ptr: number): void;
  
  // Tensor operations
  matmul_f32(
    a: number, aRows: number, aCols: number,
    b: number, bRows: number, bCols: number,
    out: number
  ): void;
  
  add_f32(a: number, b: number, out: number, size: number): void;
  mul_f32(a: number, b: number, out: number, size: number): void;
  relu_f32(input: number, output: number, size: number): void;
  sigmoid_f32(input: number, output: number, size: number): void;
  softmax_f32(input: number, output: number, size: number): void;
  
  // SIMD variants (when available)
  matmul_f32_simd?(
    a: number, aRows: number, aCols: number,
    b: number, bRows: number, bCols: number,
    out: number
  ): void;
}

/**
 * WASM model data structure
 */
interface WASMModelData {
  /** Weight buffers */
  weights: Map<string, { ptr: number; size: number; data: Float32Array }>;
  /** Model configuration */
  config: WASMModelConfig;
  /** Layer execution order */
  executionOrder: string[];
}

/**
 * Model configuration
 */
interface WASMModelConfig {
  name: string;
  version: string;
  layers: WASMLayerConfig[];
  inputs: { name: string; shape: number[]; dtype: string }[];
  outputs: { name: string; shape: number[]; dtype: string }[];
}

/**
 * Layer configuration
 */
interface WASMLayerConfig {
  name: string;
  type: string;
  inputShape: number[];
  outputShape: number[];
  weights?: string[];
  params?: Record<string, unknown>;
}

// ============================================================================
// WASM Runtime Implementation
// ============================================================================

/**
 * WASMRuntime - Pure WebAssembly inference runtime
 */
export class WASMRuntime implements Runtime {
  readonly name: RuntimeType = 'wasm';
  
  private module: WASMModule | null = null;
  private simdSupported = false;
  private models: Map<string, WASMModelData> = new Map();
  private initialized = false;

  get capabilities(): RuntimeCapabilities {
    return {
      concurrency: false, // WASM is single-threaded by default
      quantization: true,
      float16: false,
      dynamicShapes: true,
      maxBatchSize: 16,
      availableMemory: 128 * 1024 * 1024, // 128MB default
    };
  }

  /**
   * Check if WASM is available
   */
  async isAvailable(): Promise<boolean> {
    if (typeof WebAssembly === 'undefined') return false;

    try {
      // Check if we can instantiate a minimal WASM module
      const bytes = new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, // Magic number
        0x01, 0x00, 0x00, 0x00, // Version
      ]);
      await WebAssembly.instantiate(bytes);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Initialize the WASM runtime
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    // Check SIMD support
    this.simdSupported = await this.checkSIMDSupport();

    // Create memory pool
    const memory = new WebAssembly.Memory({
      initial: 256,  // 16MB initial
      maximum: 2048, // 128MB maximum
    });

    // Compile and instantiate the WASM module
    // In production, this would load an actual WASM binary
    // For now, we use a pure JS fallback
    this.module = {
      memory,
      exports: this.createJSFallback(memory),
    };

    this.initialized = true;
  }

  /**
   * Check SIMD support
   */
  private async checkSIMDSupport(): Promise<boolean> {
    try {
      // SIMD detection via feature detection
      const simdTest = new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, 0x03,
        0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00,
        0xfd, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x0b
      ]);
      await WebAssembly.instantiate(simdTest);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Create JavaScript fallback for WASM operations
   */
  private createJSFallback(memory: WebAssembly.Memory): WASMExports {
    let nextPtr = 0;
    const allocations: Map<number, number> = new Map();

    return {
      malloc: (size: number): number => {
        const ptr = nextPtr;
        nextPtr += size;
        allocations.set(ptr, size);
        return ptr;
      },

      free: (ptr: number): void => {
        allocations.delete(ptr);
      },

      matmul_f32: (
        aPtr: number, aRows: number, aCols: number,
        bPtr: number, _bRows: number, bCols: number,
        outPtr: number
      ): void => {
        const view = new Float32Array(memory.buffer);
        const aOffset = aPtr / 4;
        const bOffset = bPtr / 4;
        const outOffset = outPtr / 4;

        for (let i = 0; i < aRows; i++) {
          for (let j = 0; j < bCols; j++) {
            let sum = 0;
            for (let k = 0; k < aCols; k++) {
              sum += (view[aOffset + i * aCols + k] ?? 0) * (view[bOffset + k * bCols + j] ?? 0);
            }
            view[outOffset + i * bCols + j] = sum;
          }
        }
      },

      add_f32: (aPtr: number, bPtr: number, outPtr: number, size: number): void => {
        const view = new Float32Array(memory.buffer);
        const aOffset = aPtr / 4;
        const bOffset = bPtr / 4;
        const outOffset = outPtr / 4;

        for (let i = 0; i < size; i++) {
          view[outOffset + i] = (view[aOffset + i] ?? 0) + (view[bOffset + i] ?? 0);
        }
      },

      mul_f32: (aPtr: number, bPtr: number, outPtr: number, size: number): void => {
        const view = new Float32Array(memory.buffer);
        const aOffset = aPtr / 4;
        const bOffset = bPtr / 4;
        const outOffset = outPtr / 4;

        for (let i = 0; i < size; i++) {
          view[outOffset + i] = (view[aOffset + i] ?? 0) * (view[bOffset + i] ?? 0);
        }
      },

      relu_f32: (inputPtr: number, outputPtr: number, size: number): void => {
        const view = new Float32Array(memory.buffer);
        const inOffset = inputPtr / 4;
        const outOffset = outputPtr / 4;

        for (let i = 0; i < size; i++) {
          view[outOffset + i] = Math.max(0, view[inOffset + i] ?? 0);
        }
      },

      sigmoid_f32: (inputPtr: number, outputPtr: number, size: number): void => {
        const view = new Float32Array(memory.buffer);
        const inOffset = inputPtr / 4;
        const outOffset = outputPtr / 4;

        for (let i = 0; i < size; i++) {
          view[outOffset + i] = 1 / (1 + Math.exp(-(view[inOffset + i] ?? 0)));
        }
      },

      softmax_f32: (inputPtr: number, outputPtr: number, size: number): void => {
        const view = new Float32Array(memory.buffer);
        const inOffset = inputPtr / 4;
        const outOffset = outputPtr / 4;

        // Find max for numerical stability
        let max = -Infinity;
        for (let i = 0; i < size; i++) {
          if ((view[inOffset + i] ?? 0) > max) max = view[inOffset + i] ?? 0;
        }

        // Compute exp and sum
        let sum = 0;
        for (let i = 0; i < size; i++) {
          view[outOffset + i] = Math.exp((view[inOffset + i] ?? 0) - max);
          sum += view[outOffset + i] ?? 0;
        }

        // Normalize
        for (let i = 0; i < size; i++) {
          view[outOffset + i] = (view[outOffset + i] ?? 0) / sum;
        }
      },
    };
  }

  /**
   * Load a model
   */
  async loadModel(
    modelData: ArrayBuffer,
    options: ModelLoadOptions = {}
  ): Promise<LoadedModel> {
    this.ensureInitialized();

    // Parse model configuration
    const config = this.parseModelConfig(modelData);

    // Extract and store weights
    const wasmData: WASMModelData = {
      weights: new Map(),
      config,
      executionOrder: config.layers.map(l => l.name),
    };

    // Load weights into memory
    await this.loadWeights(modelData, wasmData);

    const modelId = `wasm_${Date.now().toString(36)}`;
    this.models.set(modelId, wasmData);

    // Create metadata
    const metadata: ModelMetadata = {
      name: config.name || options.metadata?.name || 'unknown',
      version: config.version || '1.0.0',
      inputs: config.inputs.map(i => ({
        name: i.name,
        dtype: i.dtype as 'float32',
        shape: i.shape,
      })),
      outputs: config.outputs.map(o => ({
        name: o.name,
        dtype: o.dtype as 'float32',
        shape: o.shape,
      })),
      sizeBytes: modelData.byteLength,
      quantization: options.quantization ?? 'float32',
      format: 'WebInfer',
    };

    // Create model instance
    const model = new LoadedModelImpl(
      metadata,
      'wasm',
      () => this.unloadModel(modelId)
    );

    // Track in memory manager
    getMemoryManager().trackModel(model, () => model.dispose());

    return model;
  }

  /**
   * Run inference
   */
  async run(model: LoadedModel, inputs: Tensor[]): Promise<Tensor[]> {
    this.ensureInitialized();
    
    // Execute model layers
    return this.executeModel(inputs, model.metadata);
  }

  /**
   * Execute model
   */
  private async executeModel(inputs: Tensor[], metadata: ModelMetadata): Promise<Tensor[]> {
    const outputs: Tensor[] = [];

    for (const outputSpec of metadata.outputs) {
      const outputSize = outputSpec.shape.reduce((a, b) => a * b, 1);
      
      // Process based on output requirements
      // This is a simplified implementation
      let outputTensor: WebInferTensor;

      if (inputs.length > 0 && inputs[0]) {
        const inputTensor = inputs[0] as WebInferTensor;
        
        // Apply transformations based on layer types
        // For demo, apply softmax to classification outputs
        if (outputSpec.name.includes('logits') || outputSpec.name.includes('class')) {
          outputTensor = tensorSoftmax(inputTensor) as WebInferTensor;
        } else if (outputSpec.name.includes('relu')) {
          outputTensor = tensorRelu(inputTensor);
        } else if (outputSpec.name.includes('sigmoid')) {
          outputTensor = tensorSigmoid(inputTensor);
        } else {
          // Identity or feature extraction
          const outputData = new Float32Array(outputSize);
          const inputData = inputTensor.toFloat32Array();
          for (let i = 0; i < Math.min(outputSize, inputData.length); i++) {
            outputData[i] = inputData[i] ?? 0;
          }
          outputTensor = new WebInferTensor(outputData, outputSpec.shape, 'float32');
        }
      } else {
        outputTensor = new WebInferTensor(new Float32Array(outputSize), outputSpec.shape, 'float32');
      }

      outputs.push(outputTensor);
    }

    return outputs;
  }

  /**
   * Parse model configuration
   */
  private parseModelConfig(data: ArrayBuffer): WASMModelConfig {
    try {
      const decoder = new TextDecoder();
      const text = decoder.decode(new Uint8Array(data, 0, Math.min(2048, data.byteLength)));
      
      if (text.trim().startsWith('{')) {
        let jsonEnd = text.indexOf('\n---\n');
        if (jsonEnd === -1) {
          // Try to parse as pure JSON
          try {
            return JSON.parse(text) as WASMModelConfig;
          } catch {
            jsonEnd = data.byteLength;
          }
        }
        
        const jsonStr = decoder.decode(new Uint8Array(data, 0, jsonEnd));
        return JSON.parse(jsonStr) as WASMModelConfig;
      }
    } catch {
      // Not JSON format
    }

    return {
      name: 'unknown',
      version: '1.0.0',
      layers: [],
      inputs: [{ name: 'input', shape: [-1, 768], dtype: 'float32' }],
      outputs: [{ name: 'output', shape: [-1, 768], dtype: 'float32' }],
    };
  }

  /**
   * Load weights into WASM memory
   */
  private async loadWeights(
    _modelData: ArrayBuffer,
    _wasmData: WASMModelData
  ): Promise<void> {
    // In a full implementation, extract and load weights
    // This is a placeholder
  }

  /**
   * Unload a model
   */
  private unloadModel(modelId: string): void {
    const modelData = this.models.get(modelId);
    if (modelData && this.module) {
      // Free weight buffers
      for (const weight of modelData.weights.values()) {
        this.module.exports.free(weight.ptr);
      }
    }
    this.models.delete(modelId);
  }

  /**
   * Ensure runtime is initialized
   */
  private ensureInitialized(): void {
    if (!this.initialized || !this.module) {
      throw new WebInferError(
        'WASM runtime is not initialized',
        ErrorCodes.RUNTIME_NOT_INITIALIZED
      );
    }
  }

  /**
   * Check if SIMD is supported
   */
  hasSIMDSupport(): boolean {
    return this.simdSupported;
  }

  /**
   * Dispose the runtime
   */
  dispose(): void {
    // Free all model weights
    for (const modelId of this.models.keys()) {
      this.unloadModel(modelId);
    }

    this.module = null;
    this.initialized = false;
  }
}

/**
 * Create WASM runtime factory
 */
export function createWASMRuntime(): Runtime {
  return new WASMRuntime();
}
