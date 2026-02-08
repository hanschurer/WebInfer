/**
 * WebInfer.js - Token Classification (NER) Pipeline
 *
 * Named Entity Recognition pipeline (token-classification).
 *
 * Loads a HuggingFace Hub model (default: Xenova/bert-base-NER), runs ONNX
 * inference, and merges per-token BIO tags into span entities.
 */
import { WebInferTensor } from '../core/tensor.js';
import { loadModelFromBuffer, runInference } from '../core/runtime.js';
import { fromHub, fromTask } from '../utils/hub.js';
import { loadTokenizerFromHub } from '../utils/tokenizer.js';
import { BasePipeline, registerPipeline } from './base.js';
// ============================================================================
// Pipeline
// ============================================================================
export class TokenClassificationPipeline extends BasePipeline {
    tokenizer = null;
    id2label = [];
    injectedModel = null;
    constructor(config) {
        super(config);
    }
    /**
     * Create pipeline from a pre-loaded model instance.
     * This lets apps reuse models loaded via WebInfer.loadModel(url).
     */
    static fromLoadedModel(loadedModel, tokenizer, id2label, config = { runtime: loadedModel.runtime, cache: true }) {
        const p = new TokenClassificationPipeline({
            task: 'token-classification',
            model: 'injected',
            runtime: config.runtime,
            cache: config.cache ?? true,
            quantization: config.quantization,
        });
        p.injectedModel = loadedModel;
        p.model = loadedModel;
        p.tokenizer = tokenizer;
        p.id2label = id2label;
        p.isReady = true;
        return p;
    }
    /**
     * Initialize pipeline (download model/tokenizer/config and load runtime model).
     */
    async initialize() {
        if (this.isReady && this.model && this.tokenizer)
            return;
        // If a model was injected, we're ready
        if (this.injectedModel && this.tokenizer) {
            this.model = this.injectedModel;
            this.isReady = true;
            return;
        }
        const modelSpec = this.config.model;
        // Resolve "default" to task default model on the Hub
        const bundle = modelSpec === 'default'
            ? await fromTask('token-classification', { cache: this.config.cache ?? true })
            : await fromHub(modelSpec, { cache: this.config.cache ?? true });
        // Tokenizer
        this.tokenizer = bundle.tokenizer ?? await loadTokenizerFromHub(bundle.modelId);
        // Labels (id2label) from config.json
        const id2label = bundle.config?.id2label;
        if (id2label && typeof id2label === 'object') {
            const pairs = Object.entries(id2label)
                .map(([k, v]) => [Number(k), String(v)])
                .filter(([k]) => Number.isFinite(k))
                .sort((a, b) => a[0] - b[0]);
            this.id2label = pairs.map(([, v]) => v);
        }
        else {
            // Fallback - common CoNLL tag set
            this.id2label = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'];
        }
        // Load model into runtime
        // IMPORTANT: HF models we load here are ONNX; today only the WASM(onnxruntime-web)
        // backend can execute them. If we let runtime="auto", it may pick WebGPU, which
        // currently expects WebInfer's custom GPU model format and will crash on ONNX bytes.
        const runtime = (this.config.runtime && this.config.runtime !== 'auto') ? this.config.runtime : 'wasm';
        this.model = await loadModelFromBuffer(bundle.modelData, {
            runtime,
            quantization: this.config.quantization,
            cache: this.config.cache,
            metadata: { name: bundle.modelId },
        });
        this.isReady = true;
    }
    async run(input, options) {
        await this.initialize();
        const threshold = options?.threshold ?? 0.5;
        const allowed = options?.entityTypes ? new Set(options.entityTypes) : null;
        const maxLength = options?.maxLength ?? 128;
        const encoded = this.tokenizer.encode(input, {
            maxLength,
            padding: 'max_length',
            truncation: true,
            returnTokenTypeIds: true,
        });
        // Create model input tensors (most BERT-style NER models take these 3)
        const inputIds = new WebInferTensor(encoded.inputIds, [1, encoded.inputIds.length], 'int64');
        const attentionMask = new WebInferTensor(encoded.attentionMask, [1, encoded.attentionMask.length], 'int64');
        const tokenTypeIds = new WebInferTensor(encoded.tokenTypeIds ?? new Array(encoded.inputIds.length).fill(0), [1, encoded.inputIds.length], 'int64');
        const outputs = await runInference(this.model, [inputIds, attentionMask, tokenTypeIds]);
        const entities = this.postprocessTokenClassification(outputs, input, encoded.inputIds);
        // Cleanup
        inputIds.dispose();
        attentionMask.dispose();
        tokenTypeIds.dispose();
        outputs.forEach(t => t.dispose());
        return entities
            .filter((e) => e.score >= threshold)
            .filter((e) => (allowed ? allowed.has(e.entity) : true));
    }
    // BasePipeline abstract methods (not used because we override run)
    async preprocess() {
        throw new Error('TokenClassificationPipeline.preprocess is not used');
    }
    async postprocess() {
        throw new Error('TokenClassificationPipeline.postprocess is not used');
    }
    /**
     * Convert logits -> BIO tags -> merged spans
     */
    postprocessTokenClassification(outputs, text, inputIds) {
        const logits = outputs[0];
        if (!logits)
            return [];
        const shape = logits.shape;
        const seqLen = shape[1] ?? inputIds.length;
        const numLabels = shape[2] ?? this.id2label.length;
        const data = logits.toFloat32Array();
        const offsets = this.alignOffsets(text, inputIds, seqLen);
        const toks = [];
        for (let i = 0; i < seqLen; i++) {
            const tokenStr = this.tokenizer.getToken(inputIds[i] ?? -1) ?? '';
            const off = offsets[i];
            if (!off)
                continue;
            const [start, end] = off;
            const base = i * numLabels;
            let maxIdx = 0;
            let maxLogit = data[base] ?? -Infinity;
            for (let j = 1; j < numLabels; j++) {
                const v = data[base + j] ?? -Infinity;
                if (v > maxLogit) {
                    maxLogit = v;
                    maxIdx = j;
                }
            }
            // softmax prob for maxIdx (stable)
            let sumExp = 0;
            for (let j = 0; j < numLabels; j++) {
                sumExp += Math.exp((data[base + j] ?? -Infinity) - maxLogit);
            }
            const prob = sumExp > 0 ? 1 / sumExp : 0;
            const label = this.id2label[maxIdx] ?? 'O';
            toks.push({ token: tokenStr, label, prob, start, end });
        }
        const alignedToks = toks.filter(t => t.start !== t.end);
        return mergeBioToEntities(alignedToks, text);
    }
    /**
     * Approximate token->char offset alignment for WordPiece-style tokenizers.
     * This is "good enough" for demo highlighting, and works well for BERT NER.
     */
    alignOffsets(text, inputIds, seqLen) {
        // Avoid .fill([0,0]) (shared reference)
        const offsets = Array.from({ length: seqLen }, () => [0, 0]);
        let cursor = 0;
        for (let i = 0; i < Math.min(seqLen, inputIds.length); i++) {
            const id = inputIds[i];
            const tok = this.tokenizer.getToken(id) ?? '';
            // Special tokens / padding
            if (!tok || this.tokenizer.isSpecialToken(tok) || tok === '[PAD]' || (tok.startsWith('[') && tok.endsWith(']'))) {
                offsets[i] = [0, 0];
                continue;
            }
            // Normalize token piece across common tokenizer conventions:
            // - WordPiece continuation: "##xxx"
            // - GPT2/BPE whitespace marker: "Ġxxx"
            // - SentencePiece whitespace marker: "▁xxx"
            // - Some vocabularies include leading spaces directly
            let piece = tok.startsWith('##') ? tok.slice(2) : tok;
            piece = piece.replace(/^[\s\u0120\u2581]+/g, ''); // trim spaces, Ġ, ▁
            if (!piece) {
                offsets[i] = [0, 0];
                continue;
            }
            // Find next occurrence from cursor (case-sensitive then insensitive)
            let pos = text.indexOf(piece, cursor);
            if (pos === -1)
                pos = text.toLowerCase().indexOf(piece.toLowerCase(), cursor);
            // If still not found, try to skip whitespace in source text
            if (pos === -1) {
                const nextNonSpace = text.slice(cursor).search(/\S/);
                if (nextNonSpace >= 0) {
                    const from = cursor + nextNonSpace;
                    pos = text.indexOf(piece, from);
                    if (pos === -1)
                        pos = text.toLowerCase().indexOf(piece.toLowerCase(), from);
                }
            }
            if (pos === -1) {
                offsets[i] = [0, 0];
                continue;
            }
            offsets[i] = [pos, pos + piece.length];
            cursor = pos + piece.length;
        }
        return offsets;
    }
}
function mergeBioToEntities(toks, text) {
    const entities = [];
    let curType = null;
    let curStart = 0;
    let curEnd = 0;
    let curProbSum = 0;
    let curCount = 0;
    const flush = () => {
        if (!curType || curStart === curEnd)
            return;
        const word = text.slice(curStart, curEnd);
        entities.push({
            entity: curType,
            word,
            start: curStart,
            end: curEnd,
            score: curCount ? curProbSum / curCount : 0,
        });
        curType = null;
        curStart = 0;
        curEnd = 0;
        curProbSum = 0;
        curCount = 0;
    };
    const toEntityType = (t) => {
        if (t === 'PER' || t === 'ORG' || t === 'LOC' || t === 'MISC')
            return t;
        return null;
    };
    for (const tok of toks) {
        const label = tok.label;
        if (!label || label === 'O') {
            flush();
            continue;
        }
        const m = /^(B|I)-(.+)$/.exec(label);
        if (!m) {
            flush();
            continue;
        }
        const prefix = m[1];
        const type = toEntityType(m[2]);
        if (!type) {
            flush();
            continue;
        }
        const startsNew = prefix === 'B' || curType === null || curType !== type || tok.start > curEnd + 1;
        if (startsNew) {
            flush();
            curType = type;
            curStart = tok.start;
            curEnd = tok.end;
            curProbSum = tok.prob;
            curCount = 1;
        }
        else {
            curEnd = Math.max(curEnd, tok.end);
            curProbSum += tok.prob;
            curCount += 1;
        }
    }
    flush();
    return entities;
}
// Register pipeline
registerPipeline('token-classification', (config) => new TokenClassificationPipeline(config));
//# sourceMappingURL=token-classification.js.map