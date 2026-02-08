/**
 * WebInfer.js - Memory Management
 *
 * Efficient memory management for tensors and models.
 * Features:
 * - Memory pooling
 * - Automatic garbage collection
 * - Memory tracking and statistics
 * - Leak detection
 */
/**
 * Default memory pool configuration
 */
const DEFAULT_POOL_CONFIG = {
    initialSize: 64 * 1024 * 1024, // 64MB
    maxSize: 512 * 1024 * 1024, // 512MB
    growthFactor: 1.5,
    autoGC: true,
    gcThreshold: 0.8, // 80%
};
// ============================================================================
// Memory Manager
// ============================================================================
/**
 * MemoryManager - Central memory management
 *
 * Provides:
 * - Resource tracking
 * - Memory statistics
 * - Garbage collection coordination
 * - Memory warning events
 */
export class MemoryManager {
    static instance = null;
    config;
    resources = new Map();
    disposers = new Map();
    listeners = new Map();
    allocated = 0;
    peak = 0;
    gcScheduled = false;
    disposed = false;
    constructor(config = {}) {
        this.config = { ...DEFAULT_POOL_CONFIG, ...config };
    }
    /**
     * Get singleton instance
     */
    static getInstance() {
        if (!MemoryManager.instance) {
            MemoryManager.instance = new MemoryManager();
        }
        return MemoryManager.instance;
    }
    /**
     * Configure the memory manager
     */
    static configure(config) {
        if (MemoryManager.instance) {
            console.warn('MemoryManager already initialized, configuration may not apply');
        }
        MemoryManager.instance = new MemoryManager(config);
    }
    /**
     * Track a tensor
     */
    track(tensor, disposer) {
        if (this.disposed)
            return;
        const size = this.estimateTensorSize(tensor);
        this.resources.set(tensor.id, {
            id: tensor.id,
            type: 'tensor',
            size,
            createdAt: Date.now(),
            stackTrace: this.captureStackTrace(),
        });
        if (disposer) {
            this.disposers.set(tensor.id, disposer);
        }
        this.allocated += size;
        this.peak = Math.max(this.peak, this.allocated);
        this.checkMemoryThreshold();
    }
    /**
     * Track a model
     */
    trackModel(model, disposer) {
        if (this.disposed)
            return;
        const size = model.metadata.sizeBytes;
        this.resources.set(model.id, {
            id: model.id,
            type: 'model',
            size,
            createdAt: Date.now(),
            stackTrace: this.captureStackTrace(),
        });
        if (disposer) {
            this.disposers.set(model.id, disposer);
        }
        this.allocated += size;
        this.peak = Math.max(this.peak, this.allocated);
        this.checkMemoryThreshold();
    }
    /**
     * Untrack a resource
     */
    untrack(id) {
        const resource = this.resources.get(id);
        if (resource) {
            this.allocated -= resource.size;
            this.resources.delete(id);
            this.disposers.delete(id);
        }
    }
    /**
     * Release a resource
     */
    release(resourceOrId) {
        const id = typeof resourceOrId === 'string' ? resourceOrId : resourceOrId.id;
        const disposer = this.disposers.get(id);
        if (disposer) {
            try {
                disposer();
            }
            catch (error) {
                console.error('Error disposing resource:', error);
            }
        }
        this.untrack(id);
    }
    /**
     * Estimate tensor memory size
     */
    estimateTensorSize(tensor) {
        const bytesPerElement = this.getBytesPerElement(tensor.dtype);
        return tensor.size * bytesPerElement;
    }
    /**
     * Get bytes per element for a data type
     */
    getBytesPerElement(dtype) {
        switch (dtype) {
            case 'float32':
                return 4;
            case 'float16':
                return 2;
            case 'int32':
                return 4;
            case 'int64':
                return 8;
            case 'uint8':
            case 'int8':
            case 'bool':
                return 1;
            default:
                return 4;
        }
    }
    /**
     * Capture stack trace for debugging
     */
    captureStackTrace() {
        if (typeof Error.captureStackTrace === 'function') {
            const obj = {};
            Error.captureStackTrace(obj, this.captureStackTrace);
            return obj.stack;
        }
        return new Error().stack;
    }
    /**
     * Check if memory threshold is exceeded
     */
    checkMemoryThreshold() {
        if (!this.config.autoGC)
            return;
        const usage = this.allocated / this.config.maxSize;
        if (usage >= this.config.gcThreshold && !this.gcScheduled) {
            this.gcScheduled = true;
            this.emit('memory:warning', {
                allocated: this.allocated,
                maxSize: this.config.maxSize,
                usage,
            });
            // Schedule GC on next tick
            setTimeout(() => {
                this.gc();
                this.gcScheduled = false;
            }, 0);
        }
    }
    /**
     * Garbage collection helper
     */
    gc() {
        this.emit('memory:gc', { before: this.allocated });
        // In browser environment, we can only suggest GC
        // by releasing unused resources
        // Find old resources that might be unused
        const now = Date.now();
        const oldResources = [];
        for (const [id, resource] of this.resources) {
            // Resources older than 5 minutes might be candidates for cleanup
            if (now - resource.createdAt > 5 * 60 * 1000) {
                oldResources.push(id);
            }
        }
        // Note: We don't automatically release old resources
        // This is just for reporting purposes
        // Actual cleanup should be done by the user
        this.emit('memory:gc', {
            after: this.allocated,
            potentialCleanup: oldResources.length,
        });
    }
    /**
     * Get memory statistics
     */
    getStats() {
        let tensorCount = 0;
        let modelCount = 0;
        for (const resource of this.resources.values()) {
            if (resource.type === 'tensor') {
                tensorCount++;
            }
            else {
                modelCount++;
            }
        }
        return {
            allocated: this.allocated,
            used: this.allocated, // In JS, allocated = used
            peak: this.peak,
            tensorCount,
            modelCount,
        };
    }
    /**
     * Get detailed resource list (for debugging)
     */
    getResourceDetails() {
        return Array.from(this.resources.values());
    }
    /**
     * Check for potential memory leaks
     */
    detectLeaks(maxAge = 10 * 60 * 1000) {
        const now = Date.now();
        const potentialLeaks = [];
        for (const resource of this.resources.values()) {
            if (now - resource.createdAt > maxAge) {
                potentialLeaks.push(resource);
            }
        }
        return potentialLeaks;
    }
    /**
     * Add event listener
     */
    on(event, listener) {
        let listeners = this.listeners.get(event);
        if (!listeners) {
            listeners = new Set();
            this.listeners.set(event, listeners);
        }
        listeners.add(listener);
    }
    /**
     * Remove event listener
     */
    off(event, listener) {
        const listeners = this.listeners.get(event);
        if (listeners) {
            listeners.delete(listener);
        }
    }
    /**
     * Emit event
     */
    emit(type, data) {
        const event = {
            type,
            timestamp: Date.now(),
            data,
        };
        const listeners = this.listeners.get(type);
        if (listeners) {
            for (const listener of listeners) {
                try {
                    listener(event);
                }
                catch (error) {
                    console.error('Error in event listener:', error);
                }
            }
        }
    }
    /**
     * Reset statistics
     */
    resetStats() {
        this.peak = this.allocated;
    }
    /**
     * Dispose all resources
     */
    disposeAll() {
        for (const id of this.resources.keys()) {
            this.release(id);
        }
    }
    /**
     * Dispose the manager
     */
    dispose() {
        this.disposeAll();
        this.disposed = true;
        this.listeners.clear();
        MemoryManager.instance = null;
    }
}
// ============================================================================
// Memory Scope (RAII-like pattern)
// ============================================================================
/**
 * Memory scope for automatic resource cleanup
 *
 * Usage:
 * ```typescript
 * const result = await withMemoryScope(async (scope) => {
 *   const tensor1 = scope.track(createTensor(...));
 *   const tensor2 = scope.track(createTensor(...));
 *   // Process tensors
 *   return computeResult(tensor1, tensor2);
 * });
 * // tensor1 and tensor2 are automatically disposed
 * ```
 */
export class MemoryScope {
    resources = [];
    children = [];
    parent = null;
    constructor(parent) {
        if (parent) {
            this.parent = parent;
            parent.children.push(this);
        }
    }
    /**
     * Track a resource in this scope
     */
    track(resource) {
        this.resources.push(resource);
        return resource;
    }
    /**
     * Create a child scope
     */
    createChild() {
        return new MemoryScope(this);
    }
    /**
     * Keep a resource (don't dispose it when scope ends)
     */
    keep(resource) {
        const index = this.resources.indexOf(resource);
        if (index !== -1) {
            this.resources.splice(index, 1);
        }
        return resource;
    }
    /**
     * Dispose all resources in this scope
     */
    dispose() {
        // Dispose children first
        for (const child of this.children) {
            child.dispose();
        }
        this.children = [];
        // Dispose resources in reverse order
        for (let i = this.resources.length - 1; i >= 0; i--) {
            try {
                this.resources[i]?.dispose();
            }
            catch (error) {
                console.error('Error disposing resource in scope:', error);
            }
        }
        this.resources = [];
        // Remove from parent
        if (this.parent) {
            const index = this.parent.children.indexOf(this);
            if (index !== -1) {
                this.parent.children.splice(index, 1);
            }
            this.parent = null;
        }
    }
}
/**
 * Execute a function with automatic memory cleanup
 */
export async function withMemoryScope(fn) {
    const scope = new MemoryScope();
    try {
        return await fn(scope);
    }
    finally {
        scope.dispose();
    }
}
/**
 * Synchronous version of withMemoryScope
 */
export function withMemoryScopeSync(fn) {
    const scope = new MemoryScope();
    try {
        return fn(scope);
    }
    finally {
        scope.dispose();
    }
}
// ============================================================================
// LRU Cache for Models
// ============================================================================
/**
 * LRU Cache for loaded models
 */
export class ModelCache {
    maxSize;
    maxModels;
    cache = new Map();
    currentSize = 0;
    constructor(options = {}) {
        this.maxSize = options.maxSize ?? 256 * 1024 * 1024; // 256MB default
        this.maxModels = options.maxModels ?? 5;
    }
    /**
     * Get a model from cache
     */
    get(key) {
        const entry = this.cache.get(key);
        if (entry) {
            entry.lastAccess = Date.now();
            return entry.model;
        }
        return undefined;
    }
    /**
     * Add a model to cache
     */
    set(key, model) {
        const size = model.metadata.sizeBytes;
        // Check if we need to evict
        while ((this.currentSize + size > this.maxSize || this.cache.size >= this.maxModels) &&
            this.cache.size > 0) {
            this.evictLRU();
        }
        // Add to cache
        this.cache.set(key, {
            model,
            size,
            lastAccess: Date.now(),
        });
        this.currentSize += size;
    }
    /**
     * Remove a model from cache
     */
    delete(key) {
        const entry = this.cache.get(key);
        if (entry) {
            entry.model.dispose();
            this.currentSize -= entry.size;
            this.cache.delete(key);
            return true;
        }
        return false;
    }
    /**
     * Check if model is in cache
     */
    has(key) {
        return this.cache.has(key);
    }
    /**
     * Evict least recently used model
     */
    evictLRU() {
        let oldestKey = null;
        let oldestTime = Infinity;
        for (const [key, entry] of this.cache) {
            if (entry.lastAccess < oldestTime) {
                oldestTime = entry.lastAccess;
                oldestKey = key;
            }
        }
        if (oldestKey) {
            this.delete(oldestKey);
        }
    }
    /**
     * Clear the cache
     */
    clear() {
        for (const entry of this.cache.values()) {
            entry.model.dispose();
        }
        this.cache.clear();
        this.currentSize = 0;
    }
    /**
     * Get cache statistics
     */
    getStats() {
        return {
            size: this.currentSize,
            count: this.cache.size,
            maxSize: this.maxSize,
            maxModels: this.maxModels,
        };
    }
}
// ============================================================================
// Convenience Functions
// ============================================================================
/**
 * Get memory manager instance
 */
export function getMemoryManager() {
    return MemoryManager.getInstance();
}
/**
 * Get memory statistics
 */
export function getMemoryStats() {
    return MemoryManager.getInstance().getStats();
}
/**
 * Release a resource
 */
export function release(resource) {
    MemoryManager.getInstance().release(resource);
}
/**
 * Force garbage collection hint
 */
export function gc() {
    MemoryManager.getInstance().gc();
}
//# sourceMappingURL=memory.js.map