/**
 * WebInfer.js - Benchmark Utilities
 *
 * Performance testing and comparison tools.
 */
export interface BenchmarkOptions {
    /** Number of warmup runs (default: 3) */
    warmupRuns?: number;
    /** Number of measured runs (default: 10) */
    runs?: number;
    /** Whether to log progress (default: true) */
    verbose?: boolean;
    /** Timeout per run in ms (default: 30000) */
    timeout?: number;
    /** Name for this benchmark */
    name?: string;
}
export interface BenchmarkResult {
    name: string;
    /** Average time in ms */
    avgTime: number;
    /** Median time in ms */
    medianTime: number;
    /** Minimum time in ms */
    minTime: number;
    /** Maximum time in ms */
    maxTime: number;
    /** Standard deviation in ms */
    stdDev: number;
    /** 95th percentile in ms */
    p95: number;
    /** 99th percentile in ms */
    p99: number;
    /** Throughput (ops/sec) */
    throughput: number;
    /** All individual run times */
    times: number[];
    /** Number of runs */
    totalRuns: number;
    /** Number of failed runs */
    failedRuns: number;
}
export interface CompareBenchmarkResult {
    baseline: BenchmarkResult;
    comparison: BenchmarkResult;
    speedup: number;
    percentFaster: number;
    winner: 'baseline' | 'comparison' | 'tie';
}
/**
 * Run a benchmark on an async function
 */
export declare function benchmark(fn: () => Promise<unknown> | unknown, options?: BenchmarkOptions): Promise<BenchmarkResult>;
/**
 * Compare two benchmarks
 */
export declare function compareBenchmarks(baseline: () => Promise<unknown> | unknown, comparison: () => Promise<unknown> | unknown, options?: BenchmarkOptions): Promise<CompareBenchmarkResult>;
/**
 * Run multiple benchmarks in a suite
 */
export declare function benchmarkSuite(suite: Record<string, () => Promise<unknown> | unknown>, options?: BenchmarkOptions): Promise<Record<string, BenchmarkResult>>;
/**
 * Format benchmark result as a table string
 */
export declare function formatBenchmarkResult(result: BenchmarkResult): string;
/**
 * Format comparison result
 */
export declare function formatComparisonResult(result: CompareBenchmarkResult): string;
export interface MemoryBenchmarkResult {
    name: string;
    peakMemory: number;
    avgMemory: number;
    memoryDelta: number;
}
/**
 * Benchmark memory usage
 */
export declare function benchmarkMemory(fn: () => Promise<unknown> | unknown, options?: {
    name?: string;
    runs?: number;
}): Promise<MemoryBenchmarkResult>;
declare const _default: {
    benchmark: typeof benchmark;
    compareBenchmarks: typeof compareBenchmarks;
    benchmarkSuite: typeof benchmarkSuite;
    benchmarkMemory: typeof benchmarkMemory;
    formatBenchmarkResult: typeof formatBenchmarkResult;
    formatComparisonResult: typeof formatComparisonResult;
};
export default _default;
//# sourceMappingURL=benchmark.d.ts.map