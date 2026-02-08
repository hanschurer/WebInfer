/**
 * WebInfer.js - WebNN Backend
 *
 * Browser-native neural network acceleration using the Web Neural Network API.
 * Features:
 * - Hardware-accelerated inference
 * - Native browser integration
 * - Fallback to CPU when GPU unavailable
 */
import { Runtime, RuntimeType, RuntimeCapabilities, LoadedModel, ModelLoadOptions, Tensor } from '../core/types.js';
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
declare global {
    interface Navigator {
        ml?: {
            createContext(options?: MLContextOptions): Promise<MLContext>;
        };
    }
    interface MLContext {
        compute(graph: MLGraph, inputs: Record<string, ArrayBufferView>, outputs: Record<string, ArrayBufferView>): Promise<Record<string, ArrayBufferView>>;
    }
    interface MLGraph {
    }
    interface MLGraphBuilder {
        input(name: string, desc: MLOperandDescriptor): MLOperand;
        constant(desc: MLOperandDescriptor, data: ArrayBufferView): MLOperand;
        build(outputs: Record<string, MLOperand>): Promise<MLGraph>;
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
    }
}
/**
 * WebNNRuntime - Browser-native neural network runtime
 */
export declare class WebNNRuntime implements Runtime {
    readonly name: RuntimeType;
    private context;
    private models;
    private initialized;
    private deviceType;
    get capabilities(): RuntimeCapabilities;
    /**
     * Check if WebNN is available
     */
    isAvailable(): Promise<boolean>;
    /**
     * Initialize the WebNN runtime
     */
    initialize(): Promise<void>;
    /**
     * Load a model
     */
    loadModel(modelData: ArrayBuffer, options?: ModelLoadOptions): Promise<LoadedModel>;
    /**
     * Run inference
     */
    run(model: LoadedModel, inputs: Tensor[]): Promise<Tensor[]>;
    /**
     * Execute model (simplified implementation)
     */
    private executeModel;
    /**
     * Parse model configuration
     */
    private parseModelConfig;
    /**
     * Unload a model
     */
    private unloadModel;
    /**
     * Ensure runtime is initialized
     */
    private ensureInitialized;
    /**
     * Get device type
     */
    getDeviceType(): MLContextType;
    /**
     * Dispose the runtime
     */
    dispose(): void;
}
/**
 * Create WebNN runtime factory
 */
export declare function createWebNNRuntime(): Runtime;
export {};
//# sourceMappingURL=webnn.d.ts.map