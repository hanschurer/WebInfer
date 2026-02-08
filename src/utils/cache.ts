/**
 * WebInfer.js - Caching Utilities
 * 
 * Smart caching for models, tensors, and inference results.
 */

// ============================================================================
// Cache Types
// ============================================================================

/**
 * Cache strategy types
 */
export type CacheStrategy = 'lru' | 'lfu' | 'fifo' | 'ttl';

/**
 * Cache entry
 */
interface CacheEntry<T> {
  value: T;
  size: number;
  createdAt: number;
  accessedAt: number;
  accessCount: number;
  ttl?: number;
}

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

// ============================================================================
// Cache Implementation
// ============================================================================

/**
 * Cache - Generic cache implementation
 */
export class Cache<T> {
  private readonly options: Required<CacheOptions>;
  private readonly cache: Map<string, CacheEntry<T>> = new Map();
  private currentSize = 0;
  private hits = 0;
  private misses = 0;

  constructor(options: CacheOptions = {}) {
    this.options = {
      strategy: options.strategy ?? 'lru',
      maxSize: options.maxSize ?? 100 * 1024 * 1024, // 100MB
      maxEntries: options.maxEntries ?? 1000,
      ttl: options.ttl ?? 0, // 0 = no TTL
      persistent: options.persistent ?? false,
      name: options.name ?? 'WebInfer-cache',
    };

    // Load from persistent storage if enabled
    if (this.options.persistent) {
      this.loadFromStorage();
    }
  }

  /**
   * Get value from cache
   */
  get(key: string): T | undefined {
    const entry = this.cache.get(key);
    
    if (!entry) {
      this.misses++;
      return undefined;
    }

    // Check TTL
    if (entry.ttl && Date.now() - entry.createdAt > entry.ttl) {
      this.delete(key);
      this.misses++;
      return undefined;
    }

    // Update access stats
    entry.accessedAt = Date.now();
    entry.accessCount++;
    this.hits++;

    return entry.value;
  }

  /**
   * Set value in cache
   */
  set(key: string, value: T, size: number, ttl?: number): void {
    // Remove existing entry if present
    if (this.cache.has(key)) {
      this.delete(key);
    }

    // Evict entries if necessary
    while (
      (this.currentSize + size > this.options.maxSize ||
       this.cache.size >= this.options.maxEntries) &&
      this.cache.size > 0
    ) {
      this.evict();
    }

    // Determine TTL value
    const entryTtl = ttl !== undefined ? ttl : (this.options.ttl > 0 ? this.options.ttl : undefined);

    // Add new entry
    const entry: CacheEntry<T> = {
      value,
      size,
      createdAt: Date.now(),
      accessedAt: Date.now(),
      accessCount: 1,
      ttl: entryTtl,
    };

    this.cache.set(key, entry);
    this.currentSize += size;

    // Persist if enabled
    if (this.options.persistent) {
      this.saveToStorage();
    }
  }

  /**
   * Check if key exists
   */
  has(key: string): boolean {
    const entry = this.cache.get(key);
    if (!entry) return false;

    // Check TTL
    if (entry.ttl && Date.now() - entry.createdAt > entry.ttl) {
      this.delete(key);
      return false;
    }

    return true;
  }

  /**
   * Delete entry
   */
  delete(key: string): boolean {
    const entry = this.cache.get(key);
    if (entry) {
      this.currentSize -= entry.size;
      this.cache.delete(key);
      
      if (this.options.persistent) {
        this.saveToStorage();
      }
      
      return true;
    }
    return false;
  }

  /**
   * Clear the cache
   */
  clear(): void {
    this.cache.clear();
    this.currentSize = 0;
    this.hits = 0;
    this.misses = 0;

    if (this.options.persistent) {
      this.clearStorage();
    }
  }

  /**
   * Get cache statistics
   */
  getStats(): CacheStats {
    const total = this.hits + this.misses;
    return {
      entries: this.cache.size,
      size: this.currentSize,
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? this.hits / total : 0,
    };
  }

  /**
   * Evict an entry based on strategy
   */
  private evict(): void {
    let keyToEvict: string | null = null;

    switch (this.options.strategy) {
      case 'lru':
        keyToEvict = this.findLRU();
        break;
      case 'lfu':
        keyToEvict = this.findLFU();
        break;
      case 'fifo':
        keyToEvict = this.findOldest();
        break;
      case 'ttl':
        keyToEvict = this.findExpired() ?? this.findOldest();
        break;
    }

    if (keyToEvict) {
      this.delete(keyToEvict);
    }
  }

  /**
   * Find least recently used entry
   */
  private findLRU(): string | null {
    let oldest: string | null = null;
    let oldestTime = Infinity;

    for (const [key, entry] of this.cache) {
      if (entry.accessedAt < oldestTime) {
        oldestTime = entry.accessedAt;
        oldest = key;
      }
    }

    return oldest;
  }

  /**
   * Find least frequently used entry
   */
  private findLFU(): string | null {
    let lfu: string | null = null;
    let minCount = Infinity;

    for (const [key, entry] of this.cache) {
      if (entry.accessCount < minCount) {
        minCount = entry.accessCount;
        lfu = key;
      }
    }

    return lfu;
  }

  /**
   * Find oldest entry (FIFO)
   */
  private findOldest(): string | null {
    let oldest: string | null = null;
    let oldestTime = Infinity;

    for (const [key, entry] of this.cache) {
      if (entry.createdAt < oldestTime) {
        oldestTime = entry.createdAt;
        oldest = key;
      }
    }

    return oldest;
  }

  /**
   * Find expired entry
   */
  private findExpired(): string | null {
    const now = Date.now();

    for (const [key, entry] of this.cache) {
      if (entry.ttl && now - entry.createdAt > entry.ttl) {
        return key;
      }
    }

    return null;
  }

  /**
   * Load cache from IndexedDB
   */
  private async loadFromStorage(): Promise<void> {
    if (typeof indexedDB === 'undefined') return;

    try {
      const db = await this.openDB();
      const tx = db.transaction('cache', 'readonly');
      const store = tx.objectStore('cache');
      const request = store.getAll();

      return new Promise((resolve, reject) => {
        request.onsuccess = () => {
          const entries = request.result as Array<{ key: string; entry: CacheEntry<T> }>;
          for (const { key, entry } of entries) {
            this.cache.set(key, entry);
            this.currentSize += entry.size;
          }
          resolve();
        };
        request.onerror = () => reject(request.error);
      });
    } catch {
      // Ignore storage errors
    }
  }

  /**
   * Save cache to IndexedDB
   */
  private async saveToStorage(): Promise<void> {
    if (typeof indexedDB === 'undefined') return;

    try {
      const db = await this.openDB();
      const tx = db.transaction('cache', 'readwrite');
      const store = tx.objectStore('cache');

      // Clear existing entries
      store.clear();

      // Add current entries
      for (const [key, entry] of this.cache) {
        store.put({ key, entry });
      }

      return new Promise((resolve, reject) => {
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    } catch {
      // Ignore storage errors
    }
  }

  /**
   * Clear IndexedDB storage
   */
  private async clearStorage(): Promise<void> {
    if (typeof indexedDB === 'undefined') return;

    try {
      const db = await this.openDB();
      const tx = db.transaction('cache', 'readwrite');
      const store = tx.objectStore('cache');
      store.clear();
    } catch {
      // Ignore storage errors
    }
  }

  /**
   * Open IndexedDB database
   */
  private openDB(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.options.name, 1);

      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains('cache')) {
          db.createObjectStore('cache', { keyPath: 'key' });
        }
      };

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
}

// ============================================================================
// Inference Result Cache
// ============================================================================

/**
 * InferenceCache - Cache for inference results
 */
export class InferenceCache extends Cache<Float32Array> {
  /**
   * Generate cache key from input
   */
  generateKey(modelId: string, input: Float32Array | number[]): string {
    // Create hash from input data
    const inputArray = Array.isArray(input) ? input : Array.from(input);
    const hash = this.hashArray(inputArray);
    return `${modelId}:${hash}`;
  }

  /**
   * Simple hash function for arrays
   */
  private hashArray(arr: number[]): string {
    let hash = 0;
    const sample = arr.length > 100 
      ? arr.filter((_, i) => i % Math.floor(arr.length / 100) === 0)
      : arr;
    
    for (let i = 0; i < sample.length; i++) {
      const value = sample[i] ?? 0;
      hash = ((hash << 5) - hash) + (value * 1000 | 0);
      hash |= 0;
    }
    
    return hash.toString(36);
  }
}

// ============================================================================
// Model Cache
// ============================================================================

/**
 * Model download cache using Cache API
 */
export class ModelDownloadCache {
  private readonly cacheName: string;
  private cache: globalThis.Cache | null = null;

  constructor(cacheName: string = 'WebInfer-models') {
    this.cacheName = cacheName;
  }

  /**
   * Initialize cache
   */
  private async ensureCache(): Promise<globalThis.Cache> {
    if (!this.cache) {
      if (typeof caches === 'undefined') {
        throw new Error('Cache API is not available');
      }
      this.cache = await caches.open(this.cacheName);
    }
    return this.cache;
  }

  /**
   * Get cached response
   */
  async get(url: string): Promise<Response | undefined> {
    try {
      const cache = await this.ensureCache();
      return await cache.match(url) ?? undefined;
    } catch {
      return undefined;
    }
  }

  /**
   * Store response in cache
   */
  async put(url: string, response: Response): Promise<void> {
    try {
      const cache = await this.ensureCache();
      await cache.put(url, response.clone());
    } catch {
      // Ignore cache errors
    }
  }

  /**
   * Delete cached response
   */
  async delete(url: string): Promise<boolean> {
    try {
      const cache = await this.ensureCache();
      return await cache.delete(url);
    } catch {
      return false;
    }
  }

  /**
   * Clear all cached models
   */
  async clear(): Promise<void> {
    try {
      await caches.delete(this.cacheName);
      this.cache = null;
    } catch {
      // Ignore cache errors
    }
  }

  /**
   * Get all cached URLs
   */
  async keys(): Promise<string[]> {
    try {
      const cache = await this.ensureCache();
      const requests = await cache.keys();
      return requests.map(r => r.url);
    } catch {
      return [];
    }
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a cache with common presets
 */
export function createCache<T>(
  preset: 'small' | 'medium' | 'large' | 'custom' = 'medium',
  options: CacheOptions = {}
): Cache<T> {
  const presets: Record<string, CacheOptions> = {
    small: {
      maxSize: 10 * 1024 * 1024, // 10MB
      maxEntries: 100,
    },
    medium: {
      maxSize: 100 * 1024 * 1024, // 100MB
      maxEntries: 500,
    },
    large: {
      maxSize: 500 * 1024 * 1024, // 500MB
      maxEntries: 2000,
    },
    custom: {},
  };

  return new Cache<T>({ ...presets[preset], ...options });
}
