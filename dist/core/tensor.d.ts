/**
 * WebInfer.js - Tensor Implementation
 *
 * Lightweight tensor implementation with efficient memory management.
 */
import { Tensor, DataType, Shape, TypedArray } from './types.js';
/**
 * WebInferTensor - Core tensor implementation
 */
export declare class WebInferTensor implements Tensor {
    readonly id: string;
    readonly dtype: DataType;
    readonly shape: Shape;
    readonly size: number;
    private _data;
    private _isDisposed;
    constructor(data: TypedArray | number[], shape: Shape, dtype?: DataType);
    get data(): TypedArray;
    get isDisposed(): boolean;
    /**
     * Check if tensor has been disposed
     */
    private checkDisposed;
    /**
     * Convert to Float32Array
     */
    toFloat32Array(): Float32Array;
    /**
     * Convert to regular array
     */
    toArray(): number[];
    /**
     * Clone the tensor
     */
    clone(): WebInferTensor;
    /**
     * Dispose the tensor and free memory
     */
    dispose(): void;
    /**
     * Get value at specific indices
     */
    get(...indices: number[]): number;
    /**
     * Set value at specific indices
     */
    set(value: number, ...indices: number[]): void;
    /**
     * Reshape the tensor (returns new tensor)
     */
    reshape(newShape: Shape): WebInferTensor;
    /**
     * Transpose the tensor (2D only for now)
     */
    transpose(): WebInferTensor;
    /**
     * Create string representation
     */
    toString(): string;
}
/**
 * Create a tensor from data
 */
export declare function tensor(data: TypedArray | number[] | number[][], shape?: Shape, dtype?: DataType): WebInferTensor;
/**
 * Create a tensor filled with zeros
 */
export declare function zeros(shape: Shape, dtype?: DataType): WebInferTensor;
/**
 * Create a tensor filled with ones
 */
export declare function ones(shape: Shape, dtype?: DataType): WebInferTensor;
/**
 * Create a tensor filled with a specific value
 */
export declare function full(shape: Shape, value: number, dtype?: DataType): WebInferTensor;
/**
 * Create a tensor with random values between 0 and 1
 */
export declare function random(shape: Shape, dtype?: DataType): WebInferTensor;
/**
 * Create a tensor with random values from normal distribution
 */
export declare function randn(shape: Shape, dtype?: DataType): WebInferTensor;
/**
 * Create a 1D tensor with evenly spaced values
 */
export declare function arange(start: number, stop?: number, step?: number, dtype?: DataType): WebInferTensor;
/**
 * Create a 1D tensor with evenly spaced values (specify number of points)
 */
export declare function linspace(start: number, stop: number, num?: number, dtype?: DataType): WebInferTensor;
/**
 * Create an identity matrix
 */
export declare function eye(n: number, dtype?: DataType): WebInferTensor;
/**
 * Element-wise addition
 */
export declare function add(a: WebInferTensor, b: WebInferTensor | number): WebInferTensor;
/**
 * Element-wise subtraction
 */
export declare function sub(a: WebInferTensor, b: WebInferTensor | number): WebInferTensor;
/**
 * Element-wise multiplication
 */
export declare function mul(a: WebInferTensor, b: WebInferTensor | number): WebInferTensor;
/**
 * Element-wise division
 */
export declare function div(a: WebInferTensor, b: WebInferTensor | number): WebInferTensor;
/**
 * Matrix multiplication (2D tensors)
 */
export declare function matmul(a: WebInferTensor, b: WebInferTensor): WebInferTensor;
/**
 * Softmax activation
 */
export declare function softmax(t: WebInferTensor, axis?: number): WebInferTensor;
/**
 * ReLU activation
 */
export declare function relu(t: WebInferTensor): WebInferTensor;
/**
 * Sigmoid activation
 */
export declare function sigmoid(t: WebInferTensor): WebInferTensor;
/**
 * Tanh activation
 */
export declare function tanh(t: WebInferTensor): WebInferTensor;
/**
 * Sum all elements or along an axis
 */
export declare function sum(t: WebInferTensor, axis?: number): WebInferTensor | number;
/**
 * Mean of all elements or along an axis
 */
export declare function mean(t: WebInferTensor, axis?: number): WebInferTensor | number;
/**
 * Argmax - return index of maximum value
 */
export declare function argmax(t: WebInferTensor, axis?: number): number | WebInferTensor;
/**
 * Concatenate tensors along an axis
 */
export declare function concat(tensors: WebInferTensor[], axis?: number): WebInferTensor;
//# sourceMappingURL=tensor.d.ts.map