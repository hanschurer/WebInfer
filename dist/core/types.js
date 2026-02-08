/**
 * WebInfer.js - Core Type Definitions
 *
 * This file contains all the core types used throughout the framework.
 */
// ============================================================================
// Error Types
// ============================================================================
/**
 * Base error class for WebInfer errors
 */
export class WebInferError extends Error {
    code;
    details;
    constructor(message, code, details) {
        super(message);
        this.code = code;
        this.details = details;
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
};
//# sourceMappingURL=types.js.map