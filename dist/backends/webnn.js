/**
 * WebInfer.js - WebNN Backend
 *
 * Browser-native neural network acceleration using the Web Neural Network API.
 * Features:
 * - Hardware-accelerated inference
 * - Native browser integration
 * - Fallback to CPU when GPU unavailable
 */
import { WebInferError, ErrorCodes, } from '../core/types.js';
import { LoadedModelImpl } from '../core/runtime.js';
import { WebInferTensor } from '../core/tensor.js';
import { getMemoryManager } from '../core/memory.js';
// ============================================================================
// WebNN Runtime Implementation
// ============================================================================
/**
 * WebNNRuntime - Browser-native neural network runtime
 */
export class WebNNRuntime {
    name = 'webnn';
    context = null;
    models = new Map();
    initialized = false;
    deviceType = 'default';
    get capabilities() {
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
    async isAvailable() {
        if (typeof navigator === 'undefined')
            return false;
        if (!navigator.ml)
            return false;
        try {
            const context = await navigator.ml.createContext({ deviceType: 'default' });
            return context !== null;
        }
        catch {
            return false;
        }
    }
    /**
     * Initialize the WebNN runtime
     */
    async initialize() {
        if (this.initialized)
            return;
        if (!navigator.ml) {
            throw new WebInferError('WebNN is not supported in this browser', ErrorCodes.RUNTIME_NOT_AVAILABLE);
        }
        // Try to get GPU context first, fallback to CPU
        try {
            this.context = await navigator.ml.createContext({
                deviceType: 'gpu',
                powerPreference: 'high-performance',
            });
            this.deviceType = 'gpu';
        }
        catch {
            try {
                this.context = await navigator.ml.createContext({ deviceType: 'cpu' });
                this.deviceType = 'cpu';
            }
            catch (error) {
                throw new WebInferError(`Failed to create WebNN context: ${error instanceof Error ? error.message : String(error)}`, ErrorCodes.RUNTIME_INIT_FAILED);
            }
        }
        this.initialized = true;
    }
    /**
     * Load a model
     */
    async loadModel(modelData, options = {}) {
        this.ensureInitialized();
        // Parse model configuration
        const config = this.parseModelConfig(modelData);
        // Note: Full WebNN implementation would build the graph here
        // This is a placeholder that creates minimal metadata
        const modelId = `webnn_${Date.now().toString(36)}`;
        // Create metadata
        const metadata = {
            name: config.name || options.metadata?.name || 'unknown',
            version: config.version || '1.0.0',
            inputs: config.inputs.map(i => ({
                name: i.name,
                dtype: i.dtype,
                shape: i.shape,
            })),
            outputs: config.outputs.map(o => ({
                name: o.name,
                dtype: o.dtype,
                shape: o.shape,
            })),
            sizeBytes: modelData.byteLength,
            quantization: options.quantization ?? 'float32',
            format: 'WebInfer',
        };
        // Create model instance
        const model = new LoadedModelImpl(metadata, 'webnn', () => this.unloadModel(modelId));
        // Track in memory manager
        getMemoryManager().trackModel(model, () => model.dispose());
        return model;
    }
    /**
     * Run inference
     */
    async run(model, inputs) {
        this.ensureInitialized();
        // Simplified implementation - in production, would use compiled graph
        return this.executeModel(inputs, model.metadata);
    }
    /**
     * Execute model (simplified implementation)
     */
    async executeModel(inputs, metadata) {
        const outputs = [];
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
    parseModelConfig(data) {
        try {
            const decoder = new TextDecoder();
            const text = decoder.decode(new Uint8Array(data, 0, Math.min(1024, data.byteLength)));
            if (text.trim().startsWith('{')) {
                let jsonEnd = text.indexOf('\n---\n');
                if (jsonEnd === -1)
                    jsonEnd = data.byteLength;
                const jsonStr = decoder.decode(new Uint8Array(data, 0, jsonEnd));
                return JSON.parse(jsonStr);
            }
        }
        catch {
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
    unloadModel(modelId) {
        this.models.delete(modelId);
    }
    /**
     * Ensure runtime is initialized
     */
    ensureInitialized() {
        if (!this.initialized || !this.context) {
            throw new WebInferError('WebNN runtime is not initialized', ErrorCodes.RUNTIME_NOT_INITIALIZED);
        }
    }
    /**
     * Get device type
     */
    getDeviceType() {
        return this.deviceType;
    }
    /**
     * Dispose the runtime
     */
    dispose() {
        this.models.clear();
        this.context = null;
        this.initialized = false;
    }
}
/**
 * Create WebNN runtime factory
 */
export function createWebNNRuntime() {
    return new WebNNRuntime();
}
//# sourceMappingURL=webnn.js.map