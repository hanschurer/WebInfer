/**
 * WebInfer.js - Backend Exports
 */

// WebGPU Backend
export { WebGPURuntime, createWebGPURuntime } from './webgpu.js';

// WebNN Backend
export { WebNNRuntime, createWebNNRuntime } from './webnn.js';

// WASM Backend (basic tensor ops)
export { WASMRuntime, createWASMRuntime } from './wasm.js';

// ONNX Runtime Backend (real model inference)
export { ONNXRuntime, createONNXRuntime } from './onnx.js';

// Re-export types
export type { Runtime, RuntimeType, RuntimeCapabilities } from '../core/types.js';

/**
 * Initialize all backends with the runtime manager
 */
import { registerRuntime } from '../core/runtime.js';
import { createWebGPURuntime } from './webgpu.js';
import { createWebNNRuntime } from './webnn.js';
import { createONNXRuntime } from './onnx.js';

/**
 * Register all available backends
 */
export function registerAllBackends(): void {
  registerRuntime('webgpu', createWebGPURuntime);
  registerRuntime('webnn', createWebNNRuntime);
  // Use ONNX Runtime as the WASM backend for real model inference
  registerRuntime('wasm', createONNXRuntime);
}

/**
 * Auto-register backends on module load
 */
registerAllBackends();
