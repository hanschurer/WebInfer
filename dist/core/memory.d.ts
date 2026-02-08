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
import { Tensor, LoadedModel, MemoryStats, MemoryPoolConfig, EventType, EventListener } from './types.js';
/**
 * Tracked resource info
 */
interface TrackedResource {
    id: string;
    type: 'tensor' | 'model';
    size: number;
    createdAt: number;
    stackTrace?: string;
}
/**
 * MemoryManager - Central memory management
 *
 * Provides:
 * - Resource tracking
 * - Memory statistics
 * - Garbage collection coordination
 * - Memory warning events
 */
export declare class MemoryManager {
    private static instance;
    private readonly config;
    private readonly resources;
    private readonly disposers;
    private readonly listeners;
    private allocated;
    private peak;
    private gcScheduled;
    private disposed;
    private constructor();
    /**
     * Get singleton instance
     */
    static getInstance(): MemoryManager;
    /**
     * Configure the memory manager
     */
    static configure(config: MemoryPoolConfig): void;
    /**
     * Track a tensor
     */
    track(tensor: Tensor, disposer?: () => void): void;
    /**
     * Track a model
     */
    trackModel(model: LoadedModel, disposer?: () => void): void;
    /**
     * Untrack a resource
     */
    untrack(id: string): void;
    /**
     * Release a resource
     */
    release(resourceOrId: Tensor | LoadedModel | string): void;
    /**
     * Estimate tensor memory size
     */
    private estimateTensorSize;
    /**
     * Get bytes per element for a data type
     */
    private getBytesPerElement;
    /**
     * Capture stack trace for debugging
     */
    private captureStackTrace;
    /**
     * Check if memory threshold is exceeded
     */
    private checkMemoryThreshold;
    /**
     * Garbage collection helper
     */
    gc(): void;
    /**
     * Get memory statistics
     */
    getStats(): MemoryStats;
    /**
     * Get detailed resource list (for debugging)
     */
    getResourceDetails(): TrackedResource[];
    /**
     * Check for potential memory leaks
     */
    detectLeaks(maxAge?: number): TrackedResource[];
    /**
     * Add event listener
     */
    on<T = unknown>(event: EventType, listener: EventListener<T>): void;
    /**
     * Remove event listener
     */
    off<T = unknown>(event: EventType, listener: EventListener<T>): void;
    /**
     * Emit event
     */
    private emit;
    /**
     * Reset statistics
     */
    resetStats(): void;
    /**
     * Dispose all resources
     */
    disposeAll(): void;
    /**
     * Dispose the manager
     */
    dispose(): void;
}
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
export declare class MemoryScope {
    private resources;
    private children;
    private parent;
    constructor(parent?: MemoryScope);
    /**
     * Track a resource in this scope
     */
    track<T extends {
        dispose: () => void;
    }>(resource: T): T;
    /**
     * Create a child scope
     */
    createChild(): MemoryScope;
    /**
     * Keep a resource (don't dispose it when scope ends)
     */
    keep<T extends {
        dispose: () => void;
    }>(resource: T): T;
    /**
     * Dispose all resources in this scope
     */
    dispose(): void;
}
/**
 * Execute a function with automatic memory cleanup
 */
export declare function withMemoryScope<T>(fn: (scope: MemoryScope) => Promise<T>): Promise<T>;
/**
 * Synchronous version of withMemoryScope
 */
export declare function withMemoryScopeSync<T>(fn: (scope: MemoryScope) => T): T;
/**
 * LRU Cache for loaded models
 */
export declare class ModelCache {
    private readonly maxSize;
    private readonly maxModels;
    private readonly cache;
    private currentSize;
    constructor(options?: {
        maxSize?: number;
        maxModels?: number;
    });
    /**
     * Get a model from cache
     */
    get(key: string): LoadedModel | undefined;
    /**
     * Add a model to cache
     */
    set(key: string, model: LoadedModel): void;
    /**
     * Remove a model from cache
     */
    delete(key: string): boolean;
    /**
     * Check if model is in cache
     */
    has(key: string): boolean;
    /**
     * Evict least recently used model
     */
    private evictLRU;
    /**
     * Clear the cache
     */
    clear(): void;
    /**
     * Get cache statistics
     */
    getStats(): {
        size: number;
        count: number;
        maxSize: number;
        maxModels: number;
    };
}
/**
 * Get memory manager instance
 */
export declare function getMemoryManager(): MemoryManager;
/**
 * Get memory statistics
 */
export declare function getMemoryStats(): MemoryStats;
/**
 * Release a resource
 */
export declare function release(resource: Tensor | LoadedModel): void;
/**
 * Force garbage collection hint
 */
export declare function gc(): void;
export {};
//# sourceMappingURL=memory.d.ts.map