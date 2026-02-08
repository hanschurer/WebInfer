/**
 * WebInfer.js - Caching Utilities
 *
 * Smart caching for models, tensors, and inference results.
 */
/**
 * Cache strategy types
 */
export type CacheStrategy = 'lru' | 'lfu' | 'fifo' | 'ttl';
/**
 * Cache options
 */
export interface CacheOptions {
    /** Cache strategy */
    strategy?: CacheStrategy;
    /** Maximum cache size in bytes */
    maxSize?: number;
    /** Maximum number of entries */
    maxEntries?: number;
    /** Default TTL in milliseconds */
    ttl?: number;
    /** Enable persistence to IndexedDB */
    persistent?: boolean;
    /** Cache name for persistence */
    name?: string;
}
/**
 * Cache statistics
 */
export interface CacheStats {
    /** Number of entries */
    entries: number;
    /** Total size in bytes */
    size: number;
    /** Cache hits */
    hits: number;
    /** Cache misses */
    misses: number;
    /** Hit rate (0-1) */
    hitRate: number;
}
/**
 * Cache - Generic cache implementation
 */
export declare class Cache<T> {
    private readonly options;
    private readonly cache;
    private currentSize;
    private hits;
    private misses;
    constructor(options?: CacheOptions);
    /**
     * Get value from cache
     */
    get(key: string): T | undefined;
    /**
     * Set value in cache
     */
    set(key: string, value: T, size: number, ttl?: number): void;
    /**
     * Check if key exists
     */
    has(key: string): boolean;
    /**
     * Delete entry
     */
    delete(key: string): boolean;
    /**
     * Clear the cache
     */
    clear(): void;
    /**
     * Get cache statistics
     */
    getStats(): CacheStats;
    /**
     * Evict an entry based on strategy
     */
    private evict;
    /**
     * Find least recently used entry
     */
    private findLRU;
    /**
     * Find least frequently used entry
     */
    private findLFU;
    /**
     * Find oldest entry (FIFO)
     */
    private findOldest;
    /**
     * Find expired entry
     */
    private findExpired;
    /**
     * Load cache from IndexedDB
     */
    private loadFromStorage;
    /**
     * Save cache to IndexedDB
     */
    private saveToStorage;
    /**
     * Clear IndexedDB storage
     */
    private clearStorage;
    /**
     * Open IndexedDB database
     */
    private openDB;
}
/**
 * InferenceCache - Cache for inference results
 */
export declare class InferenceCache extends Cache<Float32Array> {
    /**
     * Generate cache key from input
     */
    generateKey(modelId: string, input: Float32Array | number[]): string;
    /**
     * Simple hash function for arrays
     */
    private hashArray;
}
/**
 * Model download cache using Cache API
 */
export declare class ModelDownloadCache {
    private readonly cacheName;
    private cache;
    constructor(cacheName?: string);
    /**
     * Initialize cache
     */
    private ensureCache;
    /**
     * Get cached response
     */
    get(url: string): Promise<Response | undefined>;
    /**
     * Store response in cache
     */
    put(url: string, response: Response): Promise<void>;
    /**
     * Delete cached response
     */
    delete(url: string): Promise<boolean>;
    /**
     * Clear all cached models
     */
    clear(): Promise<void>;
    /**
     * Get all cached URLs
     */
    keys(): Promise<string[]>;
}
/**
 * Create a cache with common presets
 */
export declare function createCache<T>(preset?: 'small' | 'medium' | 'large' | 'custom', options?: CacheOptions): Cache<T>;
//# sourceMappingURL=cache.d.ts.map