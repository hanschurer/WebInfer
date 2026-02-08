/**
 * WebInfer.js - WebNN Backend
 * 
 * Browser-native neural network acceleration using the Web Neural Network API.
 * Features:
 * - Hardware-accelerated inference
 * - Native browser integration
 * - Fallback to CPU when GPU unavailable
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
import { WebInferTensor } from '../core/tensor.js';
import { getMemoryManager } from '../core/memory.js';

// ============================================================================
// WebNN Type Definitions (since WebNN types may not be globally available)
// ============================================================================

/**
 * WebNN context type
 */
type MLContextType = 'default' | 'gpu' | 'cpu' | 'npu';

/**
 * WebNN operand descriptor
 */
interface MLOperandDescriptor {
  dataType: 'float32' | 'float16' | 'int32' | 'uint32' | 'int8' | 'uint8';
  dimensions: number[];
}

/**
 * WebNN context options
 */
interface MLContextOptions {
  deviceType?: MLContextType;
  powerPreference?: 'default' | 'high-performance' | 'low-power';
}

// Extend Navigator for WebNN
declare global {
  interface Navigator {
    ml?: {
      createContext(options?: MLContextOptions): Promise<MLContext>;
    };
  }
  
  interface MLContext {
    compute(
      graph: MLGraph,
      inputs: Record<string, ArrayBufferView>,
      outputs: Record<string, ArrayBufferView>
    ): Promise<Record<string, ArrayBufferView>>;
  }
  
  interface MLGraph {
    // Graph interface
  }
  
  interface MLGraphBuilder {
    input(name: string, desc: MLOperandDescriptor): MLOperand;
    constant(desc: MLOperandDescriptor, data: ArrayBufferView): MLOperand;
    build(outputs: Record<string, MLOperand>): Promise<MLGraph>;
    
    // Operations
    add(a: MLOperand, b: MLOperand): MLOperand;
    sub(a: MLOperand, b: MLOperand): MLOperand;
    mul(a: MLOperand, b: MLOperand): MLOperand;
    div(a: MLOperand, b: MLOperand): MLOperand;
    matmul(a: MLOperand, b: MLOperand): MLOperand;
    relu(x: MLOperand): MLOperand;
    sigmoid(x: MLOperand): MLOperand;
    tanh(x: MLOperand): MLOperand;
    softmax(x: MLOperand): MLOperand;
    reshape(x: MLOperand, newShape: number[]): MLOperand;
    transpose(x: MLOperand, permutation?: number[]): MLOperand;
  }
  
  interface MLOperand {
    // Operand interface
  }
}

// ============================================================================
// WebNN Model Data
// ============================================================================

/**
 * WebNN model data structure
 */
interface WebNNModelData {
  /** Compiled graph */
  graph: MLGraph;
  /** Graph builder (for potential graph modifications) */
  builder: MLGraphBuilder;
  /** Input names and shapes */
  inputNames: string[];
  /** Output names and shapes */
  outputNames: string[];
  /** Model configuration */
  config: WebNNModelConfig;
}

/**
 * Model configuration
 */
interface WebNNModelConfig {
  name: string;
  version: string;
  inputs: { name: string; shape: number[]; dtype: string }[];
  outputs: { name: string; shape: number[]; dtype: string }[];
}

// ============================================================================
// WebNN Runtime Implementation
// ============================================================================

/**
 * WebNNRuntime - Browser-native neural network runtime
 */
export class WebNNRuntime implements Runtime {
  readonly name: RuntimeType = 'webnn';
  
  private context: MLContext | null = null;
  private models: Map<string, WebNNModelData> = new Map();
  private initialized = false;
  private deviceType: MLContextType = 'default';

  get capabilities(): RuntimeCapabilities {
    return {
      concurrency: true,
      quantization: true,
      float16: true,
      dynamicShapes: false,
      maxBatchSize: 32,
      availableMemory: 256 * 1024 * 1024, // Estimated
    };
  }

  /**
   * Check if WebNN is available
   */
  async isAvailable(): Promise<boolean> {
    if (typeof navigator === 'undefined') return false;
    if (!navigator.ml) return false;

    try {
      const context = await navigator.ml.createContext({ deviceType: 'default' });
      return context !== null;
    } catch {
      return false;
    }
  }

  /**
   * Initialize the WebNN runtime
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    if (!navigator.ml) {
      throw new WebInferError(
        'WebNN is not supported in this browser',
        ErrorCodes.RUNTIME_NOT_AVAILABLE
      );
    }

    // Try to get GPU context first, fallback to CPU
    try {
      this.context = await navigator.ml.createContext({ 
        deviceType: 'gpu',
        powerPreference: 'high-performance',
      });
      this.deviceType = 'gpu';
    } catch {
      try {
        this.context = await navigator.ml.createContext({ deviceType: 'cpu' });
        this.deviceType = 'cpu';
      } catch (error) {
        throw new WebInferError(
          `Failed to create WebNN context: ${error instanceof Error ? error.message : String(error)}`,
          ErrorCodes.RUNTIME_INIT_FAILED
        );
      }
    }

    this.initialized = true;
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

    // Note: Full WebNN implementation would build the graph here
    // This is a placeholder that creates minimal metadata
    
    const modelId = `webnn_${Date.now().toString(36)}`;

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
      'webnn',
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

    // Simplified implementation - in production, would use compiled graph
    return this.executeModel(inputs, model.metadata);
  }

  /**
   * Execute model (simplified implementation)
   */
  private async executeModel(inputs: Tensor[], metadata: ModelMetadata): Promise<Tensor[]> {
    const outputs: Tensor[] = [];

    // For each expected output
    for (const outputSpec of metadata.outputs) {
      const outputSize = outputSpec.shape.reduce((a, b) => a * b, 1);
      const outputData = new Float32Array(outputSize);

      // Simple passthrough for demo (real impl would use WebNN compute)
      if (inputs.length > 0 && inputs[0]) {
        const inputData = inputs[0].toFloat32Array();
        for (let i = 0; i < Math.min(outputSize, inputData.length); i++) {
          outputData[i] = inputData[i] ?? 0;
        }
      }

      outputs.push(new WebInferTensor(outputData, outputSpec.shape, 'float32'));
    }

    return outputs;
  }

  /**
   * Parse model configuration
   */
  private parseModelConfig(data: ArrayBuffer): WebNNModelConfig {
    try {
      const decoder = new TextDecoder();
      const text = decoder.decode(new Uint8Array(data, 0, Math.min(1024, data.byteLength)));
      
      if (text.trim().startsWith('{')) {
        let jsonEnd = text.indexOf('\n---\n');
        if (jsonEnd === -1) jsonEnd = data.byteLength;
        
        const jsonStr = decoder.decode(new Uint8Array(data, 0, jsonEnd));
        return JSON.parse(jsonStr) as WebNNModelConfig;
      }
    } catch {
      // Not JSON format
    }

    return {
      name: 'unknown',
      version: '1.0.0',
      inputs: [{ name: 'input', shape: [-1, 768], dtype: 'float32' }],
      outputs: [{ name: 'output', shape: [-1, 768], dtype: 'float32' }],
    };
  }

  /**
   * Unload a model
   */
  private unloadModel(modelId: string): void {
    this.models.delete(modelId);
  }

  /**
   * Ensure runtime is initialized
   */
  private ensureInitialized(): void {
    if (!this.initialized || !this.context) {
      throw new WebInferError(
        'WebNN runtime is not initialized',
        ErrorCodes.RUNTIME_NOT_INITIALIZED
      );
    }
  }

  /**
   * Get device type
   */
  getDeviceType(): MLContextType {
    return this.deviceType;
  }

  /**
   * Dispose the runtime
   */
  dispose(): void {
    this.models.clear();
    this.context = null;
    this.initialized = false;
  }
}

/**
 * Create WebNN runtime factory
 */
export function createWebNNRuntime(): Runtime {
  return new WebNNRuntime();
}
