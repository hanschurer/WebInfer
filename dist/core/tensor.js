/**
 * WebInfer.js - Tensor Implementation
 *
 * Lightweight tensor implementation with efficient memory management.
 */
import { WebInferError, ErrorCodes } from './types.js';
// Counter for generating unique tensor IDs
let tensorIdCounter = 0;
/**
 * Generate a unique tensor ID
 */
function generateTensorId() {
    return `tensor_${++tensorIdCounter}_${Date.now().toString(36)}`;
}
/**
 * Get the typed array constructor for a data type
 */
function getTypedArrayConstructor(dtype) {
    switch (dtype) {
        case 'float32':
            return Float32Array;
        case 'float16':
            // Float16 not natively supported, use Float32Array
            return Float32Array;
        case 'int32':
            return Int32Array;
        case 'int64':
            return BigInt64Array;
        case 'uint8':
        case 'bool':
            return Uint8Array;
        case 'int8':
            return Int8Array;
        default:
            throw new WebInferError(`Unsupported data type: ${dtype}`, ErrorCodes.INVALID_ARGUMENT, { dtype });
    }
}
/**
 * Calculate the total number of elements from shape
 */
function calculateSize(shape) {
    if (shape.length === 0)
        return 1; // Scalar
    return shape.reduce((acc, dim) => acc * dim, 1);
}
/**
 * Validate tensor shape
 */
function validateShape(shape) {
    for (let i = 0; i < shape.length; i++) {
        const dim = shape[i];
        if (dim === undefined || !Number.isInteger(dim) || dim < 0) {
            throw new WebInferError(`Invalid shape dimension at index ${i}: ${dim}`, ErrorCodes.INVALID_ARGUMENT, { shape, index: i, dimension: dim });
        }
    }
}
/**
 * WebInferTensor - Core tensor implementation
 */
export class WebInferTensor {
    id;
    dtype;
    shape;
    size;
    _data;
    _isDisposed = false;
    constructor(data, shape, dtype = 'float32') {
        validateShape(shape);
        this.id = generateTensorId();
        this.dtype = dtype;
        this.shape = Object.freeze([...shape]);
        this.size = calculateSize(this.shape);
        // Validate data size matches shape
        const expectedSize = this.size;
        if (data.length !== expectedSize) {
            throw new WebInferError(`Data length (${data.length}) does not match shape ${JSON.stringify(shape)} (expected ${expectedSize})`, ErrorCodes.TENSOR_SHAPE_MISMATCH, { dataLength: data.length, expectedSize, shape });
        }
        // Convert to appropriate typed array
        if (data instanceof Array) {
            const TypedArrayCtor = getTypedArrayConstructor(dtype);
            this._data = new TypedArrayCtor(data.length);
            if (dtype === 'int64') {
                // BigInt64Array requires BigInt values
                const bigIntData = this._data;
                for (let i = 0; i < data.length; i++) {
                    bigIntData[i] = BigInt(Math.round(data[i] ?? 0));
                }
            }
            else {
                for (let i = 0; i < data.length; i++) {
                    this._data[i] = data[i] ?? 0;
                }
            }
        }
        else {
            this._data = data;
        }
    }
    get data() {
        this.checkDisposed();
        return this._data;
    }
    get isDisposed() {
        return this._isDisposed;
    }
    /**
     * Check if tensor has been disposed
     */
    checkDisposed() {
        if (this._isDisposed) {
            throw new WebInferError('Cannot access disposed tensor', ErrorCodes.TENSOR_DISPOSED, { tensorId: this.id });
        }
    }
    /**
     * Convert to Float32Array
     */
    toFloat32Array() {
        this.checkDisposed();
        if (this._data instanceof Float32Array) {
            return this._data;
        }
        const result = new Float32Array(this.size);
        for (let i = 0; i < this.size; i++) {
            result[i] = Number(this._data[i] ?? 0);
        }
        return result;
    }
    /**
     * Convert to regular array
     */
    toArray() {
        this.checkDisposed();
        if (this.dtype === 'int64') {
            // BigInt64Array needs special handling
            const bigIntData = this._data;
            const result = [];
            for (let i = 0; i < bigIntData.length; i++) {
                result.push(Number(bigIntData[i]));
            }
            return result;
        }
        return Array.from(this._data);
    }
    /**
     * Clone the tensor
     */
    clone() {
        this.checkDisposed();
        const TypedArrayCtor = this._data.constructor;
        const clonedData = new TypedArrayCtor(this._data);
        return new WebInferTensor(clonedData, this.shape, this.dtype);
    }
    /**
     * Dispose the tensor and free memory
     */
    dispose() {
        if (!this._isDisposed) {
            this._isDisposed = true;
            // Help garbage collection - use Object.assign to avoid type issues
            Object.assign(this, { _data: null });
        }
    }
    /**
     * Get value at specific indices
     */
    get(...indices) {
        this.checkDisposed();
        if (indices.length !== this.shape.length) {
            throw new WebInferError(`Expected ${this.shape.length} indices, got ${indices.length}`, ErrorCodes.INVALID_ARGUMENT, { expectedIndices: this.shape.length, gotIndices: indices.length });
        }
        let flatIndex = 0;
        let stride = 1;
        for (let i = this.shape.length - 1; i >= 0; i--) {
            const idx = indices[i] ?? 0;
            const dim = this.shape[i] ?? 1;
            if (idx < 0 || idx >= dim) {
                throw new WebInferError(`Index ${idx} out of bounds for dimension ${i} with size ${dim}`, ErrorCodes.INVALID_ARGUMENT, { index: idx, dimension: i, size: dim });
            }
            flatIndex += idx * stride;
            stride *= dim;
        }
        return Number(this._data[flatIndex] ?? 0);
    }
    /**
     * Set value at specific indices
     */
    set(value, ...indices) {
        this.checkDisposed();
        if (indices.length !== this.shape.length) {
            throw new WebInferError(`Expected ${this.shape.length} indices, got ${indices.length}`, ErrorCodes.INVALID_ARGUMENT, { expectedIndices: this.shape.length, gotIndices: indices.length });
        }
        let flatIndex = 0;
        let stride = 1;
        for (let i = this.shape.length - 1; i >= 0; i--) {
            const idx = indices[i] ?? 0;
            const dim = this.shape[i] ?? 1;
            if (idx < 0 || idx >= dim) {
                throw new WebInferError(`Index ${idx} out of bounds for dimension ${i} with size ${dim}`, ErrorCodes.INVALID_ARGUMENT, { index: idx, dimension: i, size: dim });
            }
            flatIndex += idx * stride;
            stride *= dim;
        }
        this._data[flatIndex] = value;
    }
    /**
     * Reshape the tensor (returns new tensor)
     */
    reshape(newShape) {
        this.checkDisposed();
        const newSize = calculateSize(newShape);
        if (newSize !== this.size) {
            throw new WebInferError(`Cannot reshape tensor of size ${this.size} to shape ${JSON.stringify(newShape)} (size ${newSize})`, ErrorCodes.TENSOR_SHAPE_MISMATCH, { currentSize: this.size, newSize, newShape });
        }
        const TypedArrayCtor = this._data.constructor;
        const clonedData = new TypedArrayCtor(this._data);
        return new WebInferTensor(clonedData, newShape, this.dtype);
    }
    /**
     * Transpose the tensor (2D only for now)
     */
    transpose() {
        this.checkDisposed();
        if (this.shape.length !== 2) {
            throw new WebInferError('Transpose is currently only supported for 2D tensors', ErrorCodes.NOT_IMPLEMENTED, { shape: this.shape });
        }
        const [rows, cols] = this.shape;
        const result = new Float32Array(this.size);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result[j * rows + i] = Number(this._data[i * cols + j] ?? 0);
            }
        }
        return new WebInferTensor(result, [cols, rows], this.dtype);
    }
    /**
     * Create string representation
     */
    toString() {
        return `Tensor(shape=[${this.shape.join(', ')}], dtype=${this.dtype})`;
    }
}
// ============================================================================
// Tensor Factory Functions
// ============================================================================
/**
 * Create a tensor from data
 */
export function tensor(data, shape, dtype = 'float32') {
    // Handle nested arrays
    if (Array.isArray(data) && data.length > 0 && Array.isArray(data[0])) {
        const rows = data.length;
        const cols = data[0].length;
        const flatData = [];
        for (const row of data) {
            if (row.length !== cols) {
                throw new WebInferError('Nested arrays must have consistent dimensions', ErrorCodes.INVALID_ARGUMENT);
            }
            flatData.push(...row);
        }
        return new WebInferTensor(flatData, shape ?? [rows, cols], dtype);
    }
    // Infer shape if not provided
    const inferredShape = shape ?? [data.length];
    return new WebInferTensor(data, inferredShape, dtype);
}
/**
 * Create a tensor filled with zeros
 */
export function zeros(shape, dtype = 'float32') {
    const size = calculateSize(shape);
    const TypedArrayCtor = getTypedArrayConstructor(dtype);
    const data = new TypedArrayCtor(size);
    return new WebInferTensor(data, shape, dtype);
}
/**
 * Create a tensor filled with ones
 */
export function ones(shape, dtype = 'float32') {
    const size = calculateSize(shape);
    const TypedArrayCtor = getTypedArrayConstructor(dtype);
    const data = new TypedArrayCtor(size);
    data.fill(1);
    return new WebInferTensor(data, shape, dtype);
}
/**
 * Create a tensor filled with a specific value
 */
export function full(shape, value, dtype = 'float32') {
    const size = calculateSize(shape);
    const TypedArrayCtor = getTypedArrayConstructor(dtype);
    const data = new TypedArrayCtor(size);
    data.fill(value);
    return new WebInferTensor(data, shape, dtype);
}
/**
 * Create a tensor with random values between 0 and 1
 */
export function random(shape, dtype = 'float32') {
    const size = calculateSize(shape);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.random();
    }
    return new WebInferTensor(data, shape, dtype);
}
/**
 * Create a tensor with random values from normal distribution
 */
export function randn(shape, dtype = 'float32') {
    const size = calculateSize(shape);
    const data = new Float32Array(size);
    // Box-Muller transform for normal distribution
    for (let i = 0; i < size; i += 2) {
        const u1 = Math.random();
        const u2 = Math.random();
        const r = Math.sqrt(-2 * Math.log(u1));
        const theta = 2 * Math.PI * u2;
        data[i] = r * Math.cos(theta);
        if (i + 1 < size) {
            data[i + 1] = r * Math.sin(theta);
        }
    }
    return new WebInferTensor(data, shape, dtype);
}
/**
 * Create a 1D tensor with evenly spaced values
 */
export function arange(start, stop, step = 1, dtype = 'float32') {
    if (stop === undefined) {
        stop = start;
        start = 0;
    }
    const size = Math.ceil((stop - start) / step);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = start + i * step;
    }
    return new WebInferTensor(data, [size], dtype);
}
/**
 * Create a 1D tensor with evenly spaced values (specify number of points)
 */
export function linspace(start, stop, num = 50, dtype = 'float32') {
    const data = new Float32Array(num);
    const step = (stop - start) / (num - 1);
    for (let i = 0; i < num; i++) {
        data[i] = start + i * step;
    }
    return new WebInferTensor(data, [num], dtype);
}
/**
 * Create an identity matrix
 */
export function eye(n, dtype = 'float32') {
    const data = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
        data[i * n + i] = 1;
    }
    return new WebInferTensor(data, [n, n], dtype);
}
// ============================================================================
// Tensor Operations
// ============================================================================
/**
 * Element-wise addition
 */
export function add(a, b) {
    if (typeof b === 'number') {
        const result = new Float32Array(a.size);
        const aData = a.toFloat32Array();
        for (let i = 0; i < a.size; i++) {
            result[i] = (aData[i] ?? 0) + b;
        }
        return new WebInferTensor(result, a.shape, a.dtype);
    }
    if (a.size !== b.size) {
        throw new WebInferError('Tensor sizes must match for element-wise operations', ErrorCodes.TENSOR_SHAPE_MISMATCH, { aShape: a.shape, bShape: b.shape });
    }
    const result = new Float32Array(a.size);
    const aData = a.toFloat32Array();
    const bData = b.toFloat32Array();
    for (let i = 0; i < a.size; i++) {
        result[i] = (aData[i] ?? 0) + (bData[i] ?? 0);
    }
    return new WebInferTensor(result, a.shape, a.dtype);
}
/**
 * Element-wise subtraction
 */
export function sub(a, b) {
    if (typeof b === 'number') {
        const result = new Float32Array(a.size);
        const aData = a.toFloat32Array();
        for (let i = 0; i < a.size; i++) {
            result[i] = (aData[i] ?? 0) - b;
        }
        return new WebInferTensor(result, a.shape, a.dtype);
    }
    if (a.size !== b.size) {
        throw new WebInferError('Tensor sizes must match for element-wise operations', ErrorCodes.TENSOR_SHAPE_MISMATCH, { aShape: a.shape, bShape: b.shape });
    }
    const result = new Float32Array(a.size);
    const aData = a.toFloat32Array();
    const bData = b.toFloat32Array();
    for (let i = 0; i < a.size; i++) {
        result[i] = (aData[i] ?? 0) - (bData[i] ?? 0);
    }
    return new WebInferTensor(result, a.shape, a.dtype);
}
/**
 * Element-wise multiplication
 */
export function mul(a, b) {
    if (typeof b === 'number') {
        const result = new Float32Array(a.size);
        const aData = a.toFloat32Array();
        for (let i = 0; i < a.size; i++) {
            result[i] = (aData[i] ?? 0) * b;
        }
        return new WebInferTensor(result, a.shape, a.dtype);
    }
    if (a.size !== b.size) {
        throw new WebInferError('Tensor sizes must match for element-wise operations', ErrorCodes.TENSOR_SHAPE_MISMATCH, { aShape: a.shape, bShape: b.shape });
    }
    const result = new Float32Array(a.size);
    const aData = a.toFloat32Array();
    const bData = b.toFloat32Array();
    for (let i = 0; i < a.size; i++) {
        result[i] = (aData[i] ?? 0) * (bData[i] ?? 0);
    }
    return new WebInferTensor(result, a.shape, a.dtype);
}
/**
 * Element-wise division
 */
export function div(a, b) {
    if (typeof b === 'number') {
        const result = new Float32Array(a.size);
        const aData = a.toFloat32Array();
        for (let i = 0; i < a.size; i++) {
            result[i] = (aData[i] ?? 0) / b;
        }
        return new WebInferTensor(result, a.shape, a.dtype);
    }
    if (a.size !== b.size) {
        throw new WebInferError('Tensor sizes must match for element-wise operations', ErrorCodes.TENSOR_SHAPE_MISMATCH, { aShape: a.shape, bShape: b.shape });
    }
    const result = new Float32Array(a.size);
    const aData = a.toFloat32Array();
    const bData = b.toFloat32Array();
    for (let i = 0; i < a.size; i++) {
        result[i] = (aData[i] ?? 0) / (bData[i] ?? 0);
    }
    return new WebInferTensor(result, a.shape, a.dtype);
}
/**
 * Matrix multiplication (2D tensors)
 */
export function matmul(a, b) {
    if (a.shape.length !== 2 || b.shape.length !== 2) {
        throw new WebInferError('matmul requires 2D tensors', ErrorCodes.INVALID_ARGUMENT, { aShape: a.shape, bShape: b.shape });
    }
    const [m, k1] = a.shape;
    const [k2, n] = b.shape;
    if (k1 !== k2) {
        throw new WebInferError(`Matrix dimensions incompatible for multiplication: (${m}x${k1}) @ (${k2}x${n})`, ErrorCodes.TENSOR_SHAPE_MISMATCH, { aShape: a.shape, bShape: b.shape });
    }
    const result = new Float32Array(m * n);
    const aData = a.toFloat32Array();
    const bData = b.toFloat32Array();
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let k = 0; k < k1; k++) {
                sum += (aData[i * k1 + k] ?? 0) * (bData[k * n + j] ?? 0);
            }
            result[i * n + j] = sum;
        }
    }
    return new WebInferTensor(result, [m, n], a.dtype);
}
/**
 * Softmax activation
 */
export function softmax(t, axis = -1) {
    const data = t.toFloat32Array();
    const result = new Float32Array(t.size);
    // Handle negative axis
    const actualAxis = axis < 0 ? t.shape.length + axis : axis;
    if (actualAxis < 0 || actualAxis >= t.shape.length) {
        throw new WebInferError(`Invalid axis ${axis} for tensor with ${t.shape.length} dimensions`, ErrorCodes.INVALID_ARGUMENT, { axis, shape: t.shape });
    }
    // For 1D tensors
    if (t.shape.length === 1) {
        let max = -Infinity;
        for (let i = 0; i < t.size; i++) {
            if ((data[i] ?? 0) > max)
                max = data[i] ?? 0;
        }
        let sum = 0;
        for (let i = 0; i < t.size; i++) {
            result[i] = Math.exp((data[i] ?? 0) - max);
            sum += result[i] ?? 0;
        }
        for (let i = 0; i < t.size; i++) {
            result[i] = (result[i] ?? 0) / sum;
        }
        return new WebInferTensor(result, t.shape, t.dtype);
    }
    // For 2D tensors along last axis
    if (t.shape.length === 2 && actualAxis === 1) {
        const [rows, cols] = t.shape;
        for (let i = 0; i < rows; i++) {
            let max = -Infinity;
            for (let j = 0; j < cols; j++) {
                if ((data[i * cols + j] ?? 0) > max)
                    max = data[i * cols + j] ?? 0;
            }
            let sum = 0;
            for (let j = 0; j < cols; j++) {
                result[i * cols + j] = Math.exp((data[i * cols + j] ?? 0) - max);
                sum += result[i * cols + j] ?? 0;
            }
            for (let j = 0; j < cols; j++) {
                result[i * cols + j] = (result[i * cols + j] ?? 0) / sum;
            }
        }
        return new WebInferTensor(result, t.shape, t.dtype);
    }
    throw new WebInferError('Softmax currently only supports 1D tensors or 2D tensors along the last axis', ErrorCodes.NOT_IMPLEMENTED, { shape: t.shape, axis });
}
/**
 * ReLU activation
 */
export function relu(t) {
    const data = t.toFloat32Array();
    const result = new Float32Array(t.size);
    for (let i = 0; i < t.size; i++) {
        result[i] = Math.max(0, data[i] ?? 0);
    }
    return new WebInferTensor(result, t.shape, t.dtype);
}
/**
 * Sigmoid activation
 */
export function sigmoid(t) {
    const data = t.toFloat32Array();
    const result = new Float32Array(t.size);
    for (let i = 0; i < t.size; i++) {
        result[i] = 1 / (1 + Math.exp(-(data[i] ?? 0)));
    }
    return new WebInferTensor(result, t.shape, t.dtype);
}
/**
 * Tanh activation
 */
export function tanh(t) {
    const data = t.toFloat32Array();
    const result = new Float32Array(t.size);
    for (let i = 0; i < t.size; i++) {
        result[i] = Math.tanh(data[i] ?? 0);
    }
    return new WebInferTensor(result, t.shape, t.dtype);
}
/**
 * Sum all elements or along an axis
 */
export function sum(t, axis) {
    const data = t.toFloat32Array();
    if (axis === undefined) {
        let total = 0;
        for (let i = 0; i < t.size; i++) {
            total += data[i] ?? 0;
        }
        return total;
    }
    // Handle negative axis
    const actualAxis = axis < 0 ? t.shape.length + axis : axis;
    if (actualAxis < 0 || actualAxis >= t.shape.length) {
        throw new WebInferError(`Invalid axis ${axis} for tensor with ${t.shape.length} dimensions`, ErrorCodes.INVALID_ARGUMENT, { axis, shape: t.shape });
    }
    // Calculate new shape
    const newShape = [...t.shape];
    newShape.splice(actualAxis, 1);
    if (newShape.length === 0) {
        let total = 0;
        for (let i = 0; i < t.size; i++) {
            total += data[i] ?? 0;
        }
        return total;
    }
    // For 2D sum along axis
    if (t.shape.length === 2) {
        const [rows, cols] = t.shape;
        if (actualAxis === 0) {
            const result = new Float32Array(cols);
            for (let j = 0; j < cols; j++) {
                for (let i = 0; i < rows; i++) {
                    result[j] = (result[j] ?? 0) + (data[i * cols + j] ?? 0);
                }
            }
            return new WebInferTensor(result, [cols], t.dtype);
        }
        else {
            const result = new Float32Array(rows);
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    result[i] = (result[i] ?? 0) + (data[i * cols + j] ?? 0);
                }
            }
            return new WebInferTensor(result, [rows], t.dtype);
        }
    }
    throw new WebInferError('Sum along axis currently only supports up to 2D tensors', ErrorCodes.NOT_IMPLEMENTED, { shape: t.shape, axis });
}
/**
 * Mean of all elements or along an axis
 */
export function mean(t, axis) {
    if (axis === undefined) {
        return sum(t) / t.size;
    }
    const result = sum(t, axis);
    if (typeof result === 'number') {
        return result / (t.shape[axis] ?? 1);
    }
    const axisSize = t.shape[axis] ?? 1;
    return div(result, axisSize);
}
/**
 * Argmax - return index of maximum value
 */
export function argmax(t, axis) {
    const data = t.toFloat32Array();
    if (axis === undefined) {
        let maxIdx = 0;
        let maxVal = data[0] ?? -Infinity;
        for (let i = 1; i < t.size; i++) {
            if ((data[i] ?? -Infinity) > maxVal) {
                maxVal = data[i] ?? -Infinity;
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    // Handle negative axis
    const actualAxis = axis < 0 ? t.shape.length + axis : axis;
    // For 2D along last axis
    if (t.shape.length === 2 && actualAxis === 1) {
        const [rows, cols] = t.shape;
        const result = new Float32Array(rows);
        for (let i = 0; i < rows; i++) {
            let maxIdx = 0;
            let maxVal = data[i * cols] ?? -Infinity;
            for (let j = 1; j < cols; j++) {
                if ((data[i * cols + j] ?? -Infinity) > maxVal) {
                    maxVal = data[i * cols + j] ?? -Infinity;
                    maxIdx = j;
                }
            }
            result[i] = maxIdx;
        }
        return new WebInferTensor(result, [rows], 'int32');
    }
    throw new WebInferError('Argmax along axis currently only supports 2D tensors along the last axis', ErrorCodes.NOT_IMPLEMENTED, { shape: t.shape, axis });
}
/**
 * Concatenate tensors along an axis
 */
export function concat(tensors, axis = 0) {
    if (tensors.length === 0) {
        throw new WebInferError('Cannot concatenate empty array of tensors', ErrorCodes.INVALID_ARGUMENT);
    }
    if (tensors.length === 1) {
        return tensors[0]?.clone() ?? zeros([0]);
    }
    const first = tensors[0];
    if (!first) {
        throw new WebInferError('First tensor is undefined', ErrorCodes.INVALID_ARGUMENT);
    }
    // Handle negative axis
    const actualAxis = axis < 0 ? first.shape.length + axis : axis;
    // Validate shapes
    for (let i = 1; i < tensors.length; i++) {
        const t = tensors[i];
        if (!t)
            continue;
        if (t.shape.length !== first.shape.length) {
            throw new WebInferError('All tensors must have the same number of dimensions', ErrorCodes.TENSOR_SHAPE_MISMATCH);
        }
        for (let j = 0; j < first.shape.length; j++) {
            if (j !== actualAxis && first.shape[j] !== t.shape[j]) {
                throw new WebInferError(`Shape mismatch at dimension ${j}`, ErrorCodes.TENSOR_SHAPE_MISMATCH);
            }
        }
    }
    // Calculate new shape
    const newShape = [...first.shape];
    let totalAxisSize = 0;
    for (const t of tensors) {
        if (t)
            totalAxisSize += t.shape[actualAxis] ?? 0;
    }
    newShape[actualAxis] = totalAxisSize;
    // For 1D concatenation
    if (first.shape.length === 1) {
        const result = new Float32Array(totalAxisSize);
        let offset = 0;
        for (const t of tensors) {
            if (!t)
                continue;
            result.set(t.toFloat32Array(), offset);
            offset += t.size;
        }
        return new WebInferTensor(result, newShape, first.dtype);
    }
    throw new WebInferError('Concatenation currently only supports 1D tensors', ErrorCodes.NOT_IMPLEMENTED);
}
//# sourceMappingURL=tensor.js.map