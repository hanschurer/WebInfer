/**
 * WebInfer.js - WebGPU Backend
 *
 * High-performance WebGPU runtime for GPU-accelerated inference.
 * Features:
 * - Native concurrency support
 * - Efficient memory management
 * - Compute shader execution
 */
import { Runtime, RuntimeType, RuntimeCapabilities, LoadedModel, ModelLoadOptions, Tensor } from '../core/types.js';
declare global {
    interface Navigator {
        gpu?: GPU;
    }
    interface GPU {
        requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
    }
    interface GPURequestAdapterOptions {
        powerPreference?: 'low-power' | 'high-performance';
    }
    interface GPUAdapter {
        requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
    }
    interface GPUDeviceDescriptor {
        requiredFeatures?: string[];
        requiredLimits?: Record<string, number>;
    }
    interface GPUDevice {
        limits: GPULimits;
        lost: Promise<GPUDeviceLostInfo>;
        createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
        createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
        createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
        createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout;
        createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
        destroy(): void;
    }
    interface GPULimits {
        maxBufferSize: number;
    }
    interface GPUDeviceLostInfo {
        message: string;
        reason: string;
    }
    interface GPUBuffer {
        destroy(): void;
    }
    interface GPUShaderModule {
    }
    interface GPUBindGroupLayout {
    }
    interface GPUPipelineLayout {
    }
    interface GPUComputePipeline {
    }
    interface GPUBufferDescriptor {
        size: number;
        usage: number;
    }
    interface GPUShaderModuleDescriptor {
        code: string;
    }
    interface GPUBindGroupLayoutDescriptor {
        entries: GPUBindGroupLayoutEntry[];
    }
    interface GPUBindGroupLayoutEntry {
        binding: number;
        visibility: number;
        buffer?: {
            type: string;
        };
    }
    interface GPUPipelineLayoutDescriptor {
        bindGroupLayouts: GPUBindGroupLayout[];
    }
    interface GPUComputePipelineDescriptor {
        layout: GPUPipelineLayout;
        compute: {
            module: GPUShaderModule;
            entryPoint: string;
        };
    }
}
/**
 * WebGPURuntime - GPU-accelerated inference runtime
 */
export declare class WebGPURuntime implements Runtime {
    readonly name: RuntimeType;
    private adapter;
    private device;
    private models;
    private initialized;
    get capabilities(): RuntimeCapabilities;
    /**
     * Check if WebGPU is available
     */
    isAvailable(): Promise<boolean>;
    /**
     * Initialize the WebGPU runtime
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
     * Parse model data
     */
    private parseModelData;
    /**
     * Upload weights to GPU
     */
    private uploadWeights;
    /**
     * Create compute pipelines
     */
    private createPipelines;
    /**
     * Unload a model
     */
    private unloadModel;
    /**
     * Ensure runtime is initialized
     */
    private ensureInitialized;
    /**
     * Dispose the runtime
     */
    dispose(): void;
}
/**
 * Create WebGPU runtime factory
 */
export declare function createWebGPURuntime(): Runtime;
//# sourceMappingURL=webgpu.d.ts.map