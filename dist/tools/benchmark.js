/**
 * WebInfer.js - Benchmark Utilities
 *
 * Performance testing and comparison tools.
 */
// ============================================================================
// Benchmark Functions
// ============================================================================
/**
 * Run a benchmark on an async function
 */
export async function benchmark(fn, options = {}) {
    const { warmupRuns = 3, runs = 10, verbose = false, timeout = 30000, name = 'benchmark', } = options;
    const times = [];
    let failedRuns = 0;
    // Warmup
    if (verbose)
        console.log(`[${name}] Running ${warmupRuns} warmup iterations...`);
    for (let i = 0; i < warmupRuns; i++) {
        try {
            await Promise.race([
                Promise.resolve(fn()),
                new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), timeout)),
            ]);
        }
        catch {
            // Warmup failures are ignored
        }
    }
    // Measured runs
    if (verbose)
        console.log(`[${name}] Running ${runs} measured iterations...`);
    for (let i = 0; i < runs; i++) {
        try {
            const start = performance.now();
            await Promise.race([
                Promise.resolve(fn()),
                new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), timeout)),
            ]);
            const end = performance.now();
            times.push(end - start);
            if (verbose)
                console.log(`  Run ${i + 1}: ${(end - start).toFixed(2)}ms`);
        }
        catch (error) {
            failedRuns++;
            if (verbose)
                console.log(`  Run ${i + 1}: FAILED - ${error}`);
        }
    }
    if (times.length === 0) {
        throw new Error(`All ${runs} runs failed`);
    }
    // Calculate statistics
    const sorted = [...times].sort((a, b) => a - b);
    const sum = times.reduce((a, b) => a + b, 0);
    const avg = sum / times.length;
    const variance = times.reduce((sum, t) => sum + Math.pow(t - avg, 2), 0) / times.length;
    const stdDev = Math.sqrt(variance);
    const result = {
        name,
        avgTime: avg,
        medianTime: sorted[Math.floor(sorted.length / 2)] ?? 0,
        minTime: sorted[0] ?? 0,
        maxTime: sorted[sorted.length - 1] ?? 0,
        stdDev,
        p95: sorted[Math.floor(sorted.length * 0.95)] ?? sorted[sorted.length - 1] ?? 0,
        p99: sorted[Math.floor(sorted.length * 0.99)] ?? sorted[sorted.length - 1] ?? 0,
        throughput: 1000 / avg,
        times,
        totalRuns: runs,
        failedRuns,
    };
    if (verbose) {
        console.log(`\n[${name}] Results:`);
        console.log(`  Avg: ${result.avgTime.toFixed(2)}ms`);
        console.log(`  Median: ${result.medianTime.toFixed(2)}ms`);
        console.log(`  Min: ${result.minTime.toFixed(2)}ms`);
        console.log(`  Max: ${result.maxTime.toFixed(2)}ms`);
        console.log(`  Std Dev: ${result.stdDev.toFixed(2)}ms`);
        console.log(`  P95: ${result.p95.toFixed(2)}ms`);
        console.log(`  Throughput: ${result.throughput.toFixed(2)} ops/sec`);
    }
    return result;
}
/**
 * Compare two benchmarks
 */
export async function compareBenchmarks(baseline, comparison, options = {}) {
    const baselineResult = await benchmark(baseline, {
        ...options,
        name: options.name ? `${options.name} (baseline)` : 'baseline'
    });
    const comparisonResult = await benchmark(comparison, {
        ...options,
        name: options.name ? `${options.name} (comparison)` : 'comparison'
    });
    const speedup = baselineResult.avgTime / comparisonResult.avgTime;
    const percentFaster = ((baselineResult.avgTime - comparisonResult.avgTime) / baselineResult.avgTime) * 100;
    let winner;
    if (Math.abs(percentFaster) < 5) {
        winner = 'tie';
    }
    else if (percentFaster > 0) {
        winner = 'comparison';
    }
    else {
        winner = 'baseline';
    }
    return {
        baseline: baselineResult,
        comparison: comparisonResult,
        speedup,
        percentFaster,
        winner,
    };
}
/**
 * Run multiple benchmarks in a suite
 */
export async function benchmarkSuite(suite, options = {}) {
    const results = {};
    for (const [name, fn] of Object.entries(suite)) {
        console.log(`\n=== ${name} ===`);
        results[name] = await benchmark(fn, { ...options, name, verbose: true });
    }
    return results;
}
/**
 * Format benchmark result as a table string
 */
export function formatBenchmarkResult(result) {
    return `
┌─────────────────────────────────────────┐
│ ${result.name.padEnd(39)} │
├─────────────────────────────────────────┤
│ Avg Time:    ${result.avgTime.toFixed(2).padStart(10)}ms             │
│ Median:      ${result.medianTime.toFixed(2).padStart(10)}ms             │
│ Min Time:    ${result.minTime.toFixed(2).padStart(10)}ms             │
│ Max Time:    ${result.maxTime.toFixed(2).padStart(10)}ms             │
│ Std Dev:     ${result.stdDev.toFixed(2).padStart(10)}ms             │
│ P95:         ${result.p95.toFixed(2).padStart(10)}ms             │
│ P99:         ${result.p99.toFixed(2).padStart(10)}ms             │
│ Throughput:  ${result.throughput.toFixed(2).padStart(10)} ops/sec     │
│ Runs:        ${result.totalRuns.toString().padStart(10)} (${result.failedRuns} failed)  │
└─────────────────────────────────────────┘
  `.trim();
}
/**
 * Format comparison result
 */
export function formatComparisonResult(result) {
    const arrow = result.percentFaster > 0 ? '↑' : result.percentFaster < 0 ? '↓' : '=';
    const winnerText = result.winner === 'comparison'
        ? 'Comparison is faster!'
        : result.winner === 'baseline'
            ? 'Baseline is faster!'
            : 'Results are similar';
    return `
┌─────────────────────────────────────────────────────┐
│                  BENCHMARK COMPARISON               │
├─────────────────────────────────────────────────────┤
│ Baseline:    ${result.baseline.avgTime.toFixed(2).padStart(10)}ms                       │
│ Comparison:  ${result.comparison.avgTime.toFixed(2).padStart(10)}ms                       │
├─────────────────────────────────────────────────────┤
│ Speedup:     ${result.speedup.toFixed(2).padStart(10)}x                        │
│ Difference:  ${arrow} ${Math.abs(result.percentFaster).toFixed(1).padStart(8)}%                      │
├─────────────────────────────────────────────────────┤
│ Winner: ${winnerText.padEnd(42)} │
└─────────────────────────────────────────────────────┘
  `.trim();
}
/**
 * Benchmark memory usage
 */
export async function benchmarkMemory(fn, options = {}) {
    const { name = 'memory-benchmark', runs = 5 } = options;
    // Note: Memory APIs are limited in browsers
    // This is a simplified version that works when performance.memory is available
    const getMemory = () => {
        if (typeof performance !== 'undefined' && 'memory' in performance) {
            return performance.memory.usedJSHeapSize;
        }
        return 0;
    };
    const memoryReadings = [];
    const initialMemory = getMemory();
    for (let i = 0; i < runs; i++) {
        await fn();
        memoryReadings.push(getMemory());
    }
    const peakMemory = Math.max(...memoryReadings);
    const avgMemory = memoryReadings.reduce((a, b) => a + b, 0) / memoryReadings.length;
    const memoryDelta = avgMemory - initialMemory;
    return {
        name,
        peakMemory,
        avgMemory,
        memoryDelta,
    };
}
// ============================================================================
// Export
// ============================================================================
export default {
    benchmark,
    compareBenchmarks,
    benchmarkSuite,
    benchmarkMemory,
    formatBenchmarkResult,
    formatComparisonResult,
};
//# sourceMappingURL=benchmark.js.map