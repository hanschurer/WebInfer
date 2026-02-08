/**
 * WebInfer.js - WebAssembly Backend
 *
 * Pure WASM runtime for universal browser support.
 * Features:
 * - Universal compatibility
 * - SIMD acceleration when available
 * - Memory-efficient execution
 */
import { Runtime, RuntimeType, RuntimeCapabilities, LoadedModel, ModelLoadOptions, Tensor } from '../core/types.js';
/**
 * WASMRuntime - Pure WebAssembly inference runtime
 */
export declare class WASMRuntime implements Runtime {
    readonly name: RuntimeType;
    private module;
    private simdSupported;
    private models;
    private initialized;
    get capabilities(): RuntimeCapabilities;
    /**
     * Check if WASM is available
     */
    isAvailable(): Promise<boolean>;
    /**
     * Initialize the WASM runtime
     */
    initialize(): Promise<void>;
    /**
     * Check SIMD support
     */
    private checkSIMDSupport;
    /**
     * Create JavaScript fallback for WASM operations
     */
    private createJSFallback;
    /**
     * Load a model
     */
    loadModel(modelData: ArrayBuffer, options?: ModelLoadOptions): Promise<LoadedModel>;
    /**
     * Run inference
     */
    run(model: LoadedModel, inputs: Tensor[]): Promise<Tensor[]>;
    /**
     * Execute model
     */
    private executeModel;
    /**
     * Parse model configuration
     */
    private parseModelConfig;
    /**
     * Load weights into WASM memory
     */
    private loadWeights;
    /**
     * Unload a model
     */
    private unloadModel;
    /**
     * Ensure runtime is initialized
     */
    private ensureInitialized;
    /**
     * Check if SIMD is supported
     */
    hasSIMDSupport(): boolean;
    /**
     * Dispose the runtime
     */
    dispose(): void;
}
/**
 * Create WASM runtime factory
 */
export declare function createWASMRuntime(): Runtime;
//# sourceMappingURL=wasm.d.ts.map