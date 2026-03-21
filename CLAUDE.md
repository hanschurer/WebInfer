# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

WebInfer.js — a browser-based ML inference framework providing unified APIs across multiple compute backends (WebGPU, WebNN, WASM, ONNX Runtime). Published as `webinfer-js` on npm.

## Commands

```bash
npm run build          # TypeScript compile + browser bundle (esbuild)
npm run dev            # TypeScript watch mode
npm run lint           # ESLint on src/
npm test               # Run all tests (vitest)
npm run test:unit      # Unit tests only (tests/unit/)
npm run test:integration  # Integration tests only (tests/integration/)
npm run test:coverage  # Tests with V8 coverage
npx vitest run tests/unit/tensor.test.ts  # Run a single test file
npm run demo           # Build + start demo server at http://localhost:3000
```

## Architecture

The codebase is layered into five module groups, each with its own package.json subpath export:

- **`src/core/`** — Foundation layer. Tensor operations, task scheduler (priority queue with per-model concurrency), memory manager (scoped cleanup), runtime manager (backend selection), Web Worker pool.
- **`src/backends/`** — Compute backends. ONNX Runtime (primary, uses `onnxruntime-web`), WebGPU, WebNN, WASM. Backends self-register with RuntimeManager on import. Auto-fallback order: WebGPU → WebNN → WASM.
- **`src/pipelines/`** — High-level task APIs. `BasePipeline` abstract class with preprocess→inference→postprocess flow. 10 task pipelines (text-classification, sentiment-analysis, feature-extraction, image-classification, text-generation, token-classification, object-detection, ASR, zero-shot-classification, question-answering). Created via typed `pipeline()` factory.
- **`src/utils/`** — Tokenizers (BPE/WordPiece from HuggingFace format), image/audio preprocessors, LRU + IndexedDB model cache, model loader with resume/sharding, HuggingFace Hub client, PWA offline support.
- **`src/tools/`** — Quantization (INT8/UINT8/FLOAT16/INT4), pruning, benchmarking, tensor debugger, performance monitor.

Key patterns: singletons for RuntimeManager/MemoryManager/InferenceScheduler, factory for pipelines, registry for backends.

## TypeScript & Build

- Strict mode with all checks enabled (`noUncheckedIndexedAccess`, `noImplicitOverride`, etc.)
- ES2022 target, ESM modules (`"type": "module"`)
- Browser bundle output: `dist/WebInfer.browser.js` (via `scripts/build-browser.js` using esbuild)
- Only runtime dependency: `onnxruntime-web@^1.17.0`

## Testing

Vitest with happy-dom environment. Tests are in `tests/unit/`, `tests/integration/`, and `tests/e2e/`. Coverage excludes index.ts barrel files.

## Linting

`@typescript-eslint/no-explicit-any` is a warning (not error). Unused vars prefixed with `_` are allowed.
