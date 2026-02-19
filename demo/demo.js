/**
 * WebInfer.js Interactive Demo
 *
 * Organized into modules:
 * 1. State & Config
 * 2. Utilities
 * 3. UI Helpers
 * 4. Core Features
 * 5. Demo Class (Public API)
 * 6. Initialization
 */

import * as WebInfer from "/dist/WebInfer.browser.js";

// Expose WebInfer globally for debugging
window.WebInfer = WebInfer;

/* ==========================================================================
   1. State & Config
   ========================================================================== */

const state = {
  model: null,
  tokenizer: null,
  nerPipeline: null,
  testTensors: [],
  monitor: null,
};

const config = {
  defaultSeqLen: 128,
  monitorSampleInterval: 500,
  monitorHistorySize: 30,
};

/* ==========================================================================
   2. Utilities
   ========================================================================== */

const utils = {
  /**
   * Format bytes to human readable string
   */
  formatBytes(bytes) {
    if (!bytes) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  },

  /**
   * Sleep for given milliseconds
   */
  sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  },

  /**
   * Generate model inputs using tokenizer
   */
  createModelInputs(
    model,
    tokenizer,
    text = "Hello world",
    seqLen = config.defaultSeqLen,
  ) {
    if (tokenizer) {
      const encoded = tokenizer.encode(text, {
        maxLength: seqLen,
        padding: "max_length",
        truncation: true,
      });

      const inputIds = WebInfer.tensor(
        BigInt64Array.from(encoded.inputIds.map(BigInt)),
        [1, seqLen],
        "int64",
      );
      const attentionMask = WebInfer.tensor(
        BigInt64Array.from(encoded.attentionMask.map(BigInt)),
        [1, seqLen],
        "int64",
      );

      return [inputIds, attentionMask];
    }

    // Fallback if no tokenizer (for testing/compatibility)
    return model.metadata.inputs.map((spec) => {
      const data = new Array(seqLen).fill(0);

      if (spec.name.includes("input")) {
        data[0] = 101; // [CLS]
        data[1] = 2054; // sample token
        data[2] = 102; // [SEP]
      } else if (spec.name.includes("mask")) {
        data[0] = 1;
        data[1] = 1;
        data[2] = 1;
      }

      return WebInfer.tensor(data, [1, seqLen], "int64");
    });
  },

  /**
   * Simple tokenization and inference
   */
  async inferText(text) {
    if (!state.model) throw new Error("Model not loaded");
    if (!state.tokenizer) throw new Error("Tokenizer not loaded");

    // Use real tokenizer
    const encoded = state.tokenizer.encode(text, {
      maxLength: config.defaultSeqLen,
      padding: "max_length",
      truncation: true,
    });

    const inputs = state.model.metadata.inputs.map((spec) => {
      // Map input names to tokenizer output
      if (spec.name.includes("input_ids")) {
        return WebInfer.tensor(
          BigInt64Array.from(encoded.inputIds.map(BigInt)),
          [1, config.defaultSeqLen],
          "int64",
        );
      } else if (spec.name.includes("attention_mask")) {
        return WebInfer.tensor(
          BigInt64Array.from(encoded.attentionMask.map(BigInt)),
          [1, config.defaultSeqLen],
          "int64",
        );
      } else if (spec.name.includes("token_type_ids")) {
        return WebInfer.tensor(
          BigInt64Array.from(
            (
              encoded.tokenTypeIds || new Array(config.defaultSeqLen).fill(0)
            ).map(BigInt),
          ),
          [1, config.defaultSeqLen],
          "int64",
        );
      }
      // Fallback
      return WebInfer.tensor(
        new BigInt64Array(config.defaultSeqLen).fill(0n),
        [1, config.defaultSeqLen],
        "int64",
      );
    });

    const outputs = await WebInfer.runInference(state.model, inputs);
    const outputData = outputs[0].toArray();

    // Calculate sentiment score
    const score =
      outputData.length >= 2
        ? Math.exp(outputData[1]) /
          (Math.exp(outputData[0]) + Math.exp(outputData[1]))
        : outputData[0] > 0.5
          ? outputData[0]
          : 1 - outputData[0];

    // Cleanup
    inputs.forEach((t) => t.dispose());
    outputs.forEach((t) => t.dispose());

    return {
      label: score > 0.5 ? "positive" : "negative",
      score,
    };
  },
};

/* ==========================================================================
   3. UI Helpers
   ========================================================================== */

const ui = {
  /**
   * Get element by ID
   */
  $(id) {
    return document.getElementById(id);
  },

  /**
   * Set output content
   */
  setOutput(id, content, type = "") {
    const el = this.$(id);
    if (!el) return;

    const className = type ? `class="${type}"` : "";
    el.innerHTML = `<pre><span ${className}>${content}</span></pre>`;
  },

  /**
   * Show loading state
   */
  showLoading(id, message = "Loading...") {
    this.setOutput(id, `<span class="loader"></span>${message}`);
  },

  /**
   * Show success message
   */
  showSuccess(id, message) {
    this.setOutput(id, `✓ ${message}`, "success");
  },

  /**
   * Show error message
   */
  showError(id, error) {
    const message = error instanceof Error ? error.message : String(error);
    this.setOutput(id, `Error: ${message}`, "error");
  },

  /**
   * Render status list
   */
  renderStatusList(id, items) {
    const el = this.$(id);
    if (!el) return;

    el.innerHTML = items
      .map(
        ({ label, value, status }) => `
      <div class="status-item">
        <span>${label}</span>
        <span class="${status ? "status-badge status-" + status : ""}">${value}</span>
      </div>
    `,
      )
      .join("");
  },

  /**
   * Render metrics
   */
  renderMetrics(id, metrics) {
    const el = this.$(id);
    if (!el) return;

    el.innerHTML = metrics
      .map(
        ({ value, label }) => `
      <div class="metric">
        <div class="metric-value">${value}</div>
        <div class="metric-label">${label}</div>
      </div>
    `,
      )
      .join("");

    el.classList.remove("hidden");
  },

  /**
   * Update runtime status
   */
  async updateRuntimeStatus() {
    try {
      const runtimes = await WebInfer.getAvailableRuntimes();
      this.renderStatusList("runtime-status", [
        {
          label: "WebGPU",
          value: runtimes.get("webgpu") ? "Ready" : "N/A",
          status: runtimes.get("webgpu") ? "success" : "error",
        },
        {
          label: "WebNN",
          value: runtimes.get("webnn") ? "Ready" : "N/A",
          status: runtimes.get("webnn") ? "success" : "error",
        },
        {
          label: "WASM",
          value: runtimes.get("wasm") ? "Ready" : "N/A",
          status: runtimes.get("wasm") ? "success" : "error",
        },
      ]);
    } catch {
      this.renderStatusList("runtime-status", [
        { label: "WebGPU", value: "N/A", status: "error" },
        { label: "WebNN", value: "N/A", status: "error" },
        { label: "WASM", value: "N/A", status: "error" },
      ]);
    }
  },

  /**
   * Update memory status
   */
  updateMemoryStatus() {
    try {
      const stats = WebInfer.getMemoryStats();
      this.renderStatusList("memory-status", [
        { label: "Allocated", value: utils.formatBytes(stats.allocated || 0) },
        { label: "Peak", value: utils.formatBytes(stats.peak || 0) },
        { label: "Tensors", value: String(stats.tensorCount || 0) },
      ]);
    } catch {
      this.renderStatusList("memory-status", [
        { label: "Allocated", value: "0 B" },
        { label: "Peak", value: "0 B" },
        { label: "Tensors", value: "0" },
      ]);
    }
  },

  /**
   * Update monitor metrics
   */
  updateMonitorMetrics(sample) {
    this.renderMetrics("monitor-metrics", [
      { value: sample.inference.count, label: "Inferences" },
      { value: sample.inference.avgTime.toFixed(1) + "ms", label: "Avg Time" },
      { value: sample.inference.throughput.toFixed(1), label: "Ops/sec" },
      { value: utils.formatBytes(sample.memory.usedHeap), label: "Memory" },
      { value: sample.system.fps || "-", label: "FPS" },
    ]);
  },

  /**
   * Initialize default outputs
   */
  initOutputs() {
    const defaults = {
      "model-output": ['Click "Load Model" to download an ONNX model', "info"],
      "tensor-output": ['Click "Run Tests" to test tensor operations...', ""],
      "text-output": ["Load model first, then classify text...", ""],
      "feature-output": ["Enter text and extract features...", ""],
      "ner-output": ["Enter text and run NER...", ""],
      "quant-output": ["Test in-browser quantization...", ""],
      "debugger-output": ["Inspect tensor values and statistics...", ""],
      "benchmark-output": ["Benchmark tensor operations...", ""],
      "scheduler-output": ["Test task scheduling with priorities...", ""],
      "memory-output": ["Test memory allocation and cleanup...", ""],
      "concurrency-output": ["Test concurrent inference...", ""],
    };

    for (const [id, [msg, type]] of Object.entries(defaults)) {
      this.setOutput(id, msg, type);
    }

    // Initialize monitor metrics
    this.renderMetrics("monitor-metrics", [
      { value: "0", label: "Inferences" },
      { value: "0ms", label: "Avg Time" },
      { value: "0", label: "Ops/sec" },
      { value: "0 B", label: "Memory" },
      { value: "-", label: "FPS" },
    ]);
  },
};

/* ==========================================================================
   4. Core Features
   ========================================================================== */

const features = {
  /**
   * Load ONNX model
   */
  async loadModel() {
    const url = ui.$("model-url")?.value;
    if (!url) {
      ui.setOutput("model-output", "Enter a model URL", "warn");
      return;
    }

    ui.showLoading("model-output", "Loading model...");

    try {
      const start = performance.now();
      state.model = await WebInfer.loadModel(url, { runtime: "wasm" });

      // Attempt to load tokenizer
      // Assuming structure: .../onnx/model.onnx -> .../tokenizer.json
      let tokenizerUrl = url.replace(/onnx\/[^/]+$/, "tokenizer.json");
      if (tokenizerUrl === url) {
        // Fallback: try replacing model extension
        tokenizerUrl = url.replace(/\.onnx$/, "_tokenizer.json");
      }

      ui.showLoading("model-output", "Loading tokenizer...");
      try {
        state.tokenizer = await WebInfer.Tokenizer.fromUrl(tokenizerUrl);
      } catch (e) {
        console.warn(
          "Failed to load tokenizer from derived URL, falling back to simple tokenization",
          e,
        );
        // Fallback URL for demo (DistilBERT) if logic fails
        if (url.includes("distilbert")) {
          state.tokenizer = await WebInfer.Tokenizer.fromUrl(
            "https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/tokenizer.json",
          );
        }
      }

      const time = ((performance.now() - start) / 1000).toFixed(2);

      const info = [
        `<span class="success">✓ Model & Tokenizer loaded in ${time}s</span>`,
        `Name: ${state.model.metadata.name}`,
        `Runtime: ${state.model.runtime || "unknown"}`,
        `Size: ${utils.formatBytes(state.model.metadata.sizeBytes)}`,
        `Inputs: ${state.model.metadata.inputs.map((i) => i.name).join(", ")}`,
        `Tokenizer: ${state.tokenizer ? "Loaded" : "Not found"}`,
      ].join("\n");

      ui.$("model-output").innerHTML = `<pre>${info}</pre>`;
      ui.updateMemoryStatus();
    } catch (e) {
      ui.showError("model-output", e);
    }
  },

  /**
   * Clear loaded model
   */
  clearModel() {
    if (state.model) {
      if (state.model.dispose) state.model.dispose();
      state.model = null;
      state.tokenizer = null;
      ui.setOutput("model-output", "Model unloaded", "info");
      ui.updateMemoryStatus();
    } else {
      ui.setOutput("model-output", "No model loaded", "warn");
    }
  },

  /**
   * Clear loaded model
   */
  clearModel() {
    if (state.model) {
      if (state.model.dispose) state.model.dispose();
      state.model = null;
      state.tokenizer = null;
      ui.setOutput("model-output", "Model unloaded", "info");
      ui.updateMemoryStatus();
    } else {
      ui.setOutput("model-output", "No model loaded", "warn");
    }
  },

  /**
   * Test model inference
   */
  async testModel() {
    if (!state.model) {
      ui.setOutput("model-output", "Load model first", "warn");
      return;
    }

    ui.showLoading("model-output", "Running inference...");

    try {
      const inputs = utils.createModelInputs(state.model, state.tokenizer);
      const start = performance.now();
      const outputs = await WebInfer.runInference(state.model, inputs);
      const time = (performance.now() - start).toFixed(2);
      const data = outputs[0].toArray();

      const info = [
        `<span class="success">✓ Inference: ${time}ms</span>`,
        `Output: [${data
          .slice(0, 5)
          .map((x) => x.toFixed(4))
          .join(", ")}...]`,
      ].join("\n");

      ui.$("model-output").innerHTML = `<pre>${info}</pre>`;

      inputs.forEach((t) => t.dispose());
      outputs.forEach((t) => t.dispose());
    } catch (e) {
      ui.showError("model-output", e);
    }
  },

  /**
   * Run tensor operation tests
   */
  testTensors() {
    try {
      const a = WebInfer.tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = WebInfer.tensor([
        [5, 6],
        [7, 8],
      ]);
      const sum = WebInfer.add(a, b);
      const rand = WebInfer.random([10]);
      const probs = WebInfer.softmax(WebInfer.tensor([1, 2, 3, 4]));

      const info = [
        `<span class="success">✓ All tensor tests passed</span>`,
        `• Created 2x2 tensor`,
        `• Addition: [${sum.toArray()}]`,
        `• Random: [${rand
          .toArray()
          .slice(0, 5)
          .map((x) => x.toFixed(2))}...]`,
        `• Softmax: [${probs.toArray().map((x) => x.toFixed(3))}]`,
      ].join("\n");

      ui.$("tensor-output").innerHTML = `<pre>${info}</pre>`;

      [a, b, sum, rand, probs].forEach((t) => t.dispose());
      ui.updateMemoryStatus();
    } catch (e) {
      ui.showError("tensor-output", e);
    }
  },

  /**
   * Classify single text
   */
  async classifyText() {
    if (!state.model) {
      ui.setOutput("text-output", "Load model first", "warn");
      return;
    }

    const text = ui.$("text-input")?.value;
    if (!text) return;

    ui.showLoading("text-output", "Classifying...");

    try {
      const result = await utils.inferText(text);
      const emoji = result.label === "positive" ? "😊" : "😞";
      const pct = (result.score * 100).toFixed(1);
      ui.$("text-output").innerHTML =
        `<pre><span class="success">${emoji} ${result.label.toUpperCase()}</span> (${pct}%)</pre>`;
    } catch (e) {
      ui.showError("text-output", e);
    }
  },

  /**
   * Batch classification
   */
  async classifyBatch() {
    if (!state.model) {
      ui.setOutput("text-output", "Load model first", "warn");
      return;
    }

    const texts = [
      "I love this!",
      "This is terrible.",
      "Amazing!",
      "Worst ever.",
      "Pretty good.",
    ];
    ui.showLoading("text-output", "Processing batch...");

    try {
      const start = performance.now();
      const results = await Promise.all(texts.map((t) => utils.inferText(t)));
      const time = (performance.now() - start).toFixed(0);

      const lines = results.map((r, i) => {
        const emoji = r.label === "positive" ? "😊" : "😞";
        return `${emoji} "${texts[i]}" → ${r.label}`;
      });

      lines.push("", `<span class="success">Total: ${time}ms</span>`);
      ui.$("text-output").innerHTML = `<pre>${lines.join("\n")}</pre>`;
    } catch (e) {
      ui.showError("text-output", e);
    }
  },

  /**
   * Extract features
   */
  async extractFeatures() {
    if (!state.model) {
      ui.setOutput("feature-output", "Load model first", "warn");
      return;
    }

    const text = ui.$("feature-input")?.value;
    if (!text) return;

    ui.showLoading("feature-output", "Extracting...");

    try {
      const inputs = utils.createModelInputs(
        state.model,
        state.tokenizer,
        text,
      );
      const start = performance.now();
      const outputs = await WebInfer.runInference(state.model, inputs);
      const time = (performance.now() - start).toFixed(2);
      const embeddings = outputs[0].toArray();
      const norm = Math.sqrt(embeddings.reduce((a, b) => a + b * b, 0));

      const info = [
        `<span class="success">✓ Features extracted in ${time}ms</span>`,
        `Dimension: ${embeddings.length}`,
        `L2 Norm: ${norm.toFixed(4)}`,
        `Sample: [${embeddings
          .slice(0, 5)
          .map((x) => x.toFixed(4))
          .join(", ")}...]`,
      ].join("\n");

      ui.$("feature-output").innerHTML = `<pre>${info}</pre>`;

      inputs.forEach((t) => t.dispose());
      outputs.forEach((t) => t.dispose());
    } catch (e) {
      ui.showError("feature-output", e);
    }
  },

  /**
   * NER demo (token-classification pipeline)
   */
  async loadNERModel() {
    ui.showLoading("ner-output", "Loading NER model...");

    try {
      // Let the pipeline handle downloading model/tokenizer/config.
      state.nerPipeline = await WebInfer.pipeline("token-classification", {
        model: "default",
        runtime: "wasm",
      });

      ui.showSuccess("ner-output", "NER model ready");
    } catch (e) {
      state.nerPipeline = null;
      ui.showError("ner-output", e);
    }
  },

  async runNER() {
    const text = ui.$("ner-input")?.value;
    if (!text) return;

    if (!state.nerPipeline) {
      ui.setOutput("ner-output", "Load NER model first", "warn");
      return;
    }

    ui.showLoading("ner-output", "Running NER...");

    try {
      const ner = state.nerPipeline;

      const start = performance.now();
      const entities = await ner.run(text, { threshold: 0.5 });
      const time = (performance.now() - start).toFixed(1);

      if (!entities.length) {
        ui.$("ner-output").innerHTML =
          `<pre><span class="warn">No entities found</span>\nTime: ${time}ms</pre>`;
        return;
      }

      const lines = [
        `<span class="success">✓ Found ${entities.length} entities</span>`,
        `Time: ${time}ms`,
        "",
        ...entities.map(
          (e, i) =>
            `${i + 1}. [${e.entity}] "${e.word}" (${(e.score * 100).toFixed(1)}%) @${e.start}:${e.end}`,
        ),
      ];

      ui.$("ner-output").innerHTML = `<pre>${lines.join("\n")}</pre>`;
    } catch (e) {
      ui.showError("ner-output", e);
    }
  },

  /**
   * Quantization demo
   */
  quantize() {
    try {
      const weights = WebInfer.tensor(
        [0.5, -0.3, 0.8, -0.1, 0.9, -0.7, 0.2, -0.4],
        [2, 4],
        "float32",
      );
      const {
        tensor: quantized,
        scale,
        zeroPoint,
      } = WebInfer.quantizeTensor(weights, "int8");
      const dequantized = WebInfer.dequantizeTensor(
        quantized,
        scale,
        zeroPoint,
        "int8",
      );

      const original = weights.toArray();
      const recovered = dequantized.toArray();
      const maxError = Math.max(
        ...original.map((v, i) => Math.abs(v - recovered[i])),
      );

      const info = [
        `<span class="success">✓ Int8 Quantization</span>`,
        `Original:    [${original.map((v) => v.toFixed(3)).join(", ")}]`,
        `Quantized:   [${quantized.toArray().join(", ")}]`,
        `Dequantized: [${recovered.map((v) => v.toFixed(3)).join(", ")}]`,
        `Scale: ${scale.toFixed(6)}, Max Error: ${maxError.toFixed(6)}`,
      ].join("\n");

      ui.$("quant-output").innerHTML = `<pre>${info}</pre>`;

      [weights, quantized, dequantized].forEach((t) => t.dispose());
    } catch (e) {
      ui.showError("quant-output", e);
    }
  },

  /**
   * Pruning demo
   */
  prune() {
    try {
      const weights = WebInfer.tensor(
        [0.5, -0.1, 0.8, -0.05, 0.9, -0.02, 0.2, -0.4],
        [2, 4],
        "float32",
      );
      const { tensor: pruned, sparsity } = WebInfer.pruneTensor(weights, {
        ratio: 0.5,
      });

      const info = [
        `<span class="success">✓ Magnitude Pruning (50%)</span>`,
        `Original: [${weights
          .toArray()
          .map((v) => v.toFixed(2))
          .join(", ")}]`,
        `Pruned:   [${pruned
          .toArray()
          .map((v) => v.toFixed(2))
          .join(", ")}]`,
        `Sparsity: ${(sparsity * 100).toFixed(1)}%`,
      ].join("\n");

      ui.$("quant-output").innerHTML = `<pre>${info}</pre>`;

      [weights, pruned].forEach((t) => t.dispose());
    } catch (e) {
      ui.showError("quant-output", e);
    }
  },

  /**
   * Debugger demo
   */
  debug() {
    try {
      const data = Array.from({ length: 100 }, () => Math.random() * 2 - 1);
      const tensor = WebInfer.tensor(data, [10, 10], "float32");
      const inspection = WebInfer.inspectTensor(tensor, "random_weights");
      const histogram = WebInfer.createAsciiHistogram(
        inspection.histogram,
        25,
        4,
      );

      const info = [
        `<span class="success">Tensor: ${inspection.name}</span>`,
        `Shape: [${inspection.shape}], Size: ${inspection.size}`,
        `<span class="info">Statistics:</span>`,
        `  Min: ${inspection.stats.min.toFixed(4)}`,
        `  Max: ${inspection.stats.max.toFixed(4)}`,
        `  Mean: ${inspection.stats.mean.toFixed(4)}`,
        `  Std: ${inspection.stats.std.toFixed(4)}`,
        "",
        histogram,
      ].join("\n");

      ui.$("debugger-output").innerHTML = `<pre>${info}</pre>`;

      tensor.dispose();
    } catch (e) {
      ui.showError("debugger-output", e);
    }
  },

  /**
   * Benchmark demo
   */
  async benchmark() {
    ui.showLoading("benchmark-output", "Running benchmark...");

    try {
      const result = await WebInfer.runBenchmark(
        async () => {
          const t = WebInfer.tensor(
            Array.from({ length: 1000 }, () => Math.random()),
            [1000],
            "float32",
          );
          const sum = t.toArray().reduce((a, b) => a + b, 0);
          t.dispose();
          return sum;
        },
        { warmupRuns: 2, runs: 5, name: "Tensor Sum (1000)" },
      );

      const info = [
        `<span class="success">Benchmark: ${result.name}</span>`,
        `Avg: ${result.avgTime.toFixed(2)}ms`,
        `Min: ${result.minTime.toFixed(2)}ms`,
        `Max: ${result.maxTime.toFixed(2)}ms`,
        `Throughput: ${result.throughput.toFixed(0)} ops/sec`,
      ].join("\n");

      ui.$("benchmark-output").innerHTML = `<pre>${info}</pre>`;
    } catch (e) {
      ui.showError("benchmark-output", e);
    }
  },

  /**
   * Scheduler test
   */
  async testScheduler() {
    ui.showLoading("scheduler-output", "Testing scheduler...");

    try {
      const scheduler = WebInfer.getScheduler();
      const task1 = scheduler.schedule(
        "model-a",
        async () => {
          await utils.sleep(100);
          return "Task 1";
        },
        "high",
      );
      const task2 = scheduler.schedule(
        "model-b",
        async () => {
          await utils.sleep(50);
          return "Task 2";
        },
        "normal",
      );
      const task3 = scheduler.schedule(
        "model-a",
        async () => {
          await utils.sleep(75);
          return "Task 3";
        },
        "low",
      );

      const [r1, r2, r3] = await Promise.all([
        task1.wait(),
        task2.wait(),
        task3.wait(),
      ]);

      const info = [
        `<span class="success">✓ Scheduler Test Passed</span>`,
        `• ${r1} (high priority)`,
        `• ${r2} (normal priority)`,
        `• ${r3} (low priority)`,
      ].join("\n");

      ui.$("scheduler-output").innerHTML = `<pre>${info}</pre>`;
    } catch (e) {
      ui.showError("scheduler-output", e);
    }
  },

  /**
   * Memory allocation test
   */
  allocateMemory() {
    try {
      const before = WebInfer.getMemoryStats();

      for (let i = 0; i < 10; i++) {
        state.testTensors.push(WebInfer.random([100, 100]));
      }

      const after = WebInfer.getMemoryStats();

      const info = [
        `<span class="success">✓ Allocated 10 tensors (100x100)</span>`,
        `Before: ${utils.formatBytes(before.allocated || 0)}, ${before.tensorCount || 0} tensors`,
        `After: ${utils.formatBytes(after.allocated || 0)}, ${after.tensorCount || 0} tensors`,
      ].join("\n");

      ui.$("memory-output").innerHTML = `<pre>${info}</pre>`;
      ui.updateMemoryStatus();
    } catch (e) {
      ui.showError("memory-output", e);
    }
  },

  /**
   * Memory cleanup
   */
  cleanupMemory() {
    state.testTensors.forEach((t) => {
      if (!t.isDisposed) t.dispose();
    });
    state.testTensors = [];
    WebInfer.gc();

    ui.showSuccess("memory-output", "Memory cleaned up");
    ui.updateMemoryStatus();
  },

  /**
   * Concurrency test
   */
  async testConcurrency() {
    if (!state.model) {
      ui.setOutput("concurrency-output", "Load model first", "warn");
      ui.$("concurrency-metrics")?.classList.add("hidden");
      return;
    }

    ui.showLoading("concurrency-output", "Running concurrent tasks...");

    try {
      const texts = [
        "Great!",
        "Terrible!",
        "Amazing!",
        "Awful!",
        "Good!",
        "Bad!",
        "Nice!",
        "Horrible!",
      ];
      const start = performance.now();
      const results = await Promise.all(texts.map((t) => utils.inferText(t)));
      const total = performance.now() - start;

      const lines = [
        `<span class="success">✓ Concurrent execution complete</span>`,
        ...results.map(
          (r, i) => `${r.label === "positive" ? "😊" : "😞"} "${texts[i]}"`,
        ),
      ];

      ui.$("concurrency-output").innerHTML = `<pre>${lines.join("\n")}</pre>`;

      ui.renderMetrics("concurrency-metrics", [
        { value: total.toFixed(0) + "ms", label: "Total" },
        { value: String(texts.length), label: "Tasks" },
        { value: (total / texts.length).toFixed(0) + "ms", label: "Avg" },
      ]);
    } catch (e) {
      ui.showError("concurrency-output", e);
    }
  },

  /**
   * Start performance monitor
   */
  startMonitor() {
    if (!state.monitor) {
      state.monitor = new WebInfer.PerformanceMonitor({
        sampleInterval: config.monitorSampleInterval,
        historySize: config.monitorHistorySize,
      });
      state.monitor.onSample((sample) => ui.updateMonitorMetrics(sample));
    }
    state.monitor.start();
  },

  /**
   * Stop monitor
   */
  stopMonitor() {
    if (state.monitor) {
      state.monitor.stop();
    }
  },

  /**
   * Simulate inferences for monitor
   */
  simulateInferences() {
    if (!state.monitor) {
      this.startMonitor();
    }

    for (let i = 0; i < 5; i++) {
      setTimeout(() => {
        state.monitor?.recordInference(30 + Math.random() * 70);
      }, i * 100);
    }
  },

  /**
   * Open dashboard modal
   */
  openDashboard() {
    if (!state.monitor) {
      this.startMonitor();
      this.simulateInferences();
    }

    const modal = ui.$("dashboard-modal");
    const frame = ui.$("dashboard-frame");

    if (modal && frame) {
      frame.srcdoc = WebInfer.generateDashboardHTML(state.monitor);
      modal.classList.add("active");
      document.body.style.overflow = "hidden";
    }
  },

  /**
   * Close dashboard modal
   */
  closeDashboard() {
    const modal = ui.$("dashboard-modal");
    if (modal) {
      modal.classList.remove("active");
      document.body.style.overflow = "";
    }
  },
};

/* ==========================================================================
   5. Demo Class (Public API)
   ========================================================================== */

/**
 * Demo public API - exposed to window for onclick handlers
 */
window.Demo = {
  // Model
  loadModel: () => features.loadModel(),
  testModel: () => features.testModel(),
  clearModel: () => features.clearModel(),

  // Core
  testTensors: () => features.testTensors(),
  classifyText: () => features.classifyText(),
  classifyBatch: () => features.classifyBatch(),
  extractFeatures: () => features.extractFeatures(),
  loadNERModel: () => features.loadNERModel(),
  runNER: () => features.runNER(),

  // Tools
  quantize: () => features.quantize(),
  prune: () => features.prune(),
  debug: () => features.debug(),
  benchmark: () => features.benchmark(),

  // System
  testScheduler: () => features.testScheduler(),
  allocateMemory: () => features.allocateMemory(),
  cleanupMemory: () => features.cleanupMemory(),
  testConcurrency: () => features.testConcurrency(),

  // Monitor
  startMonitor: () => features.startMonitor(),
  stopMonitor: () => features.stopMonitor(),
  simulateInferences: () => features.simulateInferences(),
  openDashboard: () => features.openDashboard(),
  closeDashboard: () => features.closeDashboard(),
};

/* ==========================================================================
   6. Initialization
   ========================================================================== */

/**
 * Initialize demo on DOM ready
 */
async function init() {
  // Initialize UI
  ui.initOutputs();
  await ui.updateRuntimeStatus();
  ui.updateMemoryStatus();

  // Setup modal close handlers
  const modal = ui.$("dashboard-modal");
  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        features.closeDashboard();
      }
    });
  }

  // ESC key closes modal
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      features.closeDashboard();
    }
  });

  console.log("✓ WebInfer.js Demo initialized");
}

// Wait for DOM
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
