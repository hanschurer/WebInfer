/**
 * WebInfer.js - Text Generation Pipeline
 *
 * Autoregressive text generation with streaming support.
 * Supports GPT-2, LLaMA, Mistral, and other causal LM models.
 */
import { BasePipeline, PipelineResult } from './base.js';
import { Tokenizer } from '../utils/tokenizer.js';
import { WebInferTensor } from '../core/tensor.js';
import { PipelineConfig, PipelineOptions } from '../core/types.js';
/**
 * Text generation options
 */
export interface TextGenerationOptions {
    /** Maximum number of new tokens to generate */
    maxNewTokens?: number;
    /** Maximum total length (prompt + generated) */
    maxLength?: number;
    /** Minimum number of new tokens to generate */
    minNewTokens?: number;
    /** Sampling temperature (higher = more random) */
    temperature?: number;
    /** Top-k sampling (0 = disabled) */
    topK?: number;
    /** Top-p (nucleus) sampling (1.0 = disabled) */
    topP?: number;
    /** Repetition penalty (1.0 = disabled) */
    repetitionPenalty?: number;
    /** Stop sequences */
    stopSequences?: string[];
    /** Whether to do sampling (false = greedy) */
    doSample?: boolean;
    /** Number of sequences to return */
    numReturnSequences?: number;
    /** Return full text (including prompt) */
    returnFullText?: boolean;
    /** Callback for each generated token */
    onToken?: (token: string, tokenId: number) => void;
}
/**
 * Text generation result
 */
export interface TextGenerationResult extends PipelineResult {
    /** Generated text */
    generatedText: string;
    /** Full text (prompt + generated) if returnFullText is true */
    fullText?: string;
    /** Generated token IDs */
    tokenIds: number[];
    /** Number of tokens generated */
    numTokens: number;
}
/**
 * Streaming generation event
 */
export interface GenerationStreamEvent {
    /** Current token */
    token: string;
    /** Token ID */
    tokenId: number;
    /** Generated text so far */
    generatedText: string;
    /** Whether generation is complete */
    done: boolean;
}
/**
 * TextGenerationPipeline - Autoregressive text generation
 *
 * @example
 * ```typescript
 * const generator = await pipeline('text-generation', 'Xenova/gpt2');
 *
 * // Simple generation
 * const result = await generator.run('Once upon a time');
 * console.log(result.generatedText);
 *
 * // Streaming generation
 * for await (const event of generator.stream('Hello, ')) {
 *   process.stdout.write(event.token);
 * }
 * ```
 */
export declare class TextGenerationPipeline extends BasePipeline<string | string[], TextGenerationResult | TextGenerationResult[]> {
    private tokenizer;
    private eosTokenId;
    constructor(config?: PipelineConfig);
    /**
     * Set tokenizer
     */
    setTokenizer(tokenizer: Tokenizer): void;
    /**
     * Preprocess - not used for text generation (handled in generateSingle)
     */
    protected preprocess(input: string | string[]): Promise<WebInferTensor[]>;
    /**
     * Postprocess - not used for text generation (handled in generateSingle)
     */
    protected postprocess(_outputs: WebInferTensor[], _options?: PipelineOptions): Promise<TextGenerationResult | TextGenerationResult[]>;
    /**
     * Generate text (non-streaming)
     */
    run(prompt: string | string[], options?: PipelineOptions & TextGenerationOptions): Promise<TextGenerationResult | TextGenerationResult[]>;
    /**
     * Generate text with streaming (async generator)
     */
    stream(prompt: string, options?: TextGenerationOptions): AsyncGenerator<GenerationStreamEvent>;
    /**
     * Generate a single sequence (non-streaming)
     */
    private generateSingle;
    /**
     * Generate next token using the model
     */
    private generateNextToken;
    /**
     * Greedy decoding (argmax)
     */
    private greedy;
    /**
     * Sample from probability distribution with top-k/top-p filtering
     */
    private sample;
}
/**
 * Create text generation pipeline
 */
export declare function createTextGenerationPipeline(config?: PipelineConfig): TextGenerationPipeline;
//# sourceMappingURL=text-generation.d.ts.map