/**
 * WebInfer.js - Token Classification (NER) Pipeline
 *
 * Named Entity Recognition pipeline (token-classification).
 *
 * Loads a HuggingFace Hub model (default: Xenova/bert-base-NER), runs ONNX
 * inference, and merges per-token BIO tags into span entities.
 */
import { LoadedModel, PipelineConfig, PipelineOptions } from '../core/types.js';
import { Tokenizer } from '../utils/tokenizer.js';
import { BasePipeline, type PipelineResult } from './base.js';
export type NerEntityType = 'PER' | 'ORG' | 'LOC' | 'MISC';
export interface NerEntity extends PipelineResult {
    /** Entity type */
    entity: NerEntityType;
    /** Entity text */
    word: string;
    /** Confidence score (0-1) */
    score: number;
    /** Start character offset (inclusive) */
    start: number;
    /** End character offset (exclusive) */
    end: number;
}
export interface TokenClassificationOptions extends PipelineOptions {
    /** Minimum score threshold (default: 0.5) */
    threshold?: number;
    /** Restrict entity types */
    entityTypes?: NerEntityType[];
    /** Max sequence length (default: 128) */
    maxLength?: number;
}
export declare class TokenClassificationPipeline extends BasePipeline<string, NerEntity[]> {
    private tokenizer;
    private id2label;
    private injectedModel;
    constructor(config: PipelineConfig);
    /**
     * Create pipeline from a pre-loaded model instance.
     * This lets apps reuse models loaded via WebInfer.loadModel(url).
     */
    static fromLoadedModel(loadedModel: LoadedModel, tokenizer: Tokenizer, id2label: string[], config?: Omit<PipelineConfig, 'task' | 'model'>): TokenClassificationPipeline;
    /**
     * Initialize pipeline (download model/tokenizer/config and load runtime model).
     */
    initialize(): Promise<void>;
    run(input: string, options?: TokenClassificationOptions): Promise<NerEntity[]>;
    protected preprocess(): Promise<never>;
    protected postprocess(): Promise<never>;
    /**
     * Convert logits -> BIO tags -> merged spans
     */
    private postprocessTokenClassification;
    /**
     * Approximate token->char offset alignment for WordPiece-style tokenizers.
     * This is "good enough" for demo highlighting, and works well for BERT NER.
     */
    private alignOffsets;
}
//# sourceMappingURL=token-classification.d.ts.map