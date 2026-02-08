/**
 * WebInfer.js - Backend Exports
 */
export { WebGPURuntime, createWebGPURuntime } from './webgpu.js';
export { WebNNRuntime, createWebNNRuntime } from './webnn.js';
export { WASMRuntime, createWASMRuntime } from './wasm.js';
export { ONNXRuntime, createONNXRuntime } from './onnx.js';
export type { Runtime, RuntimeType, RuntimeCapabilities } from '../core/types.js';
/**
 * Register all available backends
 */
export declare function registerAllBackends(): void;
//# sourceMappingURL=index.d.ts.map