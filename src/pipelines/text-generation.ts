/**
 * WebInfer.js - Text Generation Pipeline
 * 
 * Autoregressive text generation with streaming support.
 * Supports GPT-2, LLaMA, Mistral, and other causal LM models.
 */

import { BasePipeline, PipelineResult } from './base.js';
import { Tokenizer } from '../utils/tokenizer.js';
import { WebInferTensor, softmax } from '../core/tensor.js';
import { PipelineConfig, PipelineOptions } from '../core/types.js';
import { runInference } from '../core/runtime.js';

// ============================================================================
// Types
// ============================================================================

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

// ============================================================================
// Text Generation Pipeline
// ============================================================================

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
export class TextGenerationPipeline extends BasePipeline<string | string[], TextGenerationResult | TextGenerationResult[]> {
  private tokenizer: Tokenizer | null = null;
  private eosTokenId: number = 50256; // GPT-2 default

  constructor(config?: PipelineConfig) {
    super(config ?? {
      task: 'text-generation',
      model: 'default',
    });
  }

  /**
   * Set tokenizer
   */
  setTokenizer(tokenizer: Tokenizer): void {
    this.tokenizer = tokenizer;
    const specialIds = tokenizer.getSpecialTokenIds();
    this.eosTokenId = specialIds.eosTokenId ?? specialIds.sepTokenId ?? 50256;
  }

  /**
   * Preprocess - not used for text generation (handled in generateSingle)
   */
  protected async preprocess(input: string | string[]): Promise<WebInferTensor[]> {
    // For text generation, preprocessing is handled in generateNextToken
    const text = Array.isArray(input) ? input[0] ?? '' : input;
    if (!this.tokenizer) {
      // Return dummy tensor if no tokenizer
      return [new WebInferTensor(new Float32Array([0]), [1], 'float32')];
    }
    const encoded = this.tokenizer.encode(text, {
      addSpecialTokens: false,
      padding: 'do_not_pad',
    });
    return [new WebInferTensor(
      BigInt64Array.from(encoded.inputIds.map(id => BigInt(id))),
      [1, encoded.inputIds.length],
      'int64'
    )];
  }

  /**
   * Postprocess - not used for text generation (handled in generateSingle)
   */
  protected async postprocess(
    _outputs: WebInferTensor[],
    _options?: PipelineOptions
  ): Promise<TextGenerationResult | TextGenerationResult[]> {
    // For text generation, postprocessing is handled in generateSingle
    return {
      generatedText: '',
      tokenIds: [],
      numTokens: 0,
      processingTime: 0,
    };
  }

  /**
   * Generate text (non-streaming)
   */
  override async run(
    prompt: string | string[],
    options?: PipelineOptions & TextGenerationOptions
  ): Promise<TextGenerationResult | TextGenerationResult[]> {
    await this.initialize();
    
    const prompts = Array.isArray(prompt) ? prompt : [prompt];
    const results = await Promise.all(
      prompts.map(p => this.generateSingle(p, options ?? {}))
    );
    return Array.isArray(prompt) ? results : results[0]!;
  }

  /**
   * Generate text with streaming (async generator)
   */
  async *stream(
    prompt: string,
    options: TextGenerationOptions = {}
  ): AsyncGenerator<GenerationStreamEvent> {
    const startTime = performance.now();
    
    if (!this.tokenizer) {
      throw new Error('Tokenizer not set. Call setTokenizer() first.');
    }

    const {
      maxNewTokens = 50,
      maxLength = 512,
      temperature = 1.0,
      topK = 0,
      topP = 1.0,
      repetitionPenalty = 1.0,
      stopSequences = [],
      doSample = true,
    } = options;

    // Encode prompt
    const encoded = this.tokenizer.encode(prompt, {
      addSpecialTokens: false,
      padding: 'do_not_pad',
      truncation: false,
    });

    let inputIds = [...encoded.inputIds];
    const generatedIds: number[] = [];
    let generatedText = '';

    // Generation loop
    for (let i = 0; i < maxNewTokens; i++) {
      // Check max length
      if (inputIds.length >= maxLength) break;

      // Run model forward pass
      const nextTokenId = await this.generateNextToken(
        inputIds,
        temperature,
        topK,
        topP,
        repetitionPenalty,
        doSample
      );

      // Check for EOS
      if (nextTokenId === this.eosTokenId) {
        yield {
          token: '',
          tokenId: nextTokenId,
          generatedText,
          done: true,
        };
        break;
      }

      // Decode token
      const token = this.tokenizer.decode([nextTokenId], true);
      generatedIds.push(nextTokenId);
      inputIds.push(nextTokenId);
      generatedText += token;

      // Call token callback
      if (options.onToken) {
        options.onToken(token, nextTokenId);
      }

      // Check stop sequences
      let shouldStop = false;
      for (const stopSeq of stopSequences) {
        if (generatedText.endsWith(stopSeq)) {
          generatedText = generatedText.slice(0, -stopSeq.length);
          shouldStop = true;
          break;
        }
      }

      yield {
        token,
        tokenId: nextTokenId,
        generatedText,
        done: shouldStop,
      };

      if (shouldStop) break;
    }

    // Final event
    const endTime = performance.now();
    console.log(`Generation completed in ${(endTime - startTime).toFixed(2)}ms`);
  }

  /**
   * Generate a single sequence (non-streaming)
   */
  private async generateSingle(
    prompt: string,
    options: TextGenerationOptions
  ): Promise<TextGenerationResult> {
    const startTime = performance.now();
    
    if (!this.tokenizer) {
      throw new Error('Tokenizer not set. Call setTokenizer() first.');
    }

    const {
      maxNewTokens = 50,
      maxLength = 512,
      temperature = 1.0,
      topK = 0,
      topP = 1.0,
      repetitionPenalty = 1.0,
      stopSequences = [],
      doSample = true,
      returnFullText = false,
    } = options;

    // Encode prompt
    const encoded = this.tokenizer.encode(prompt, {
      addSpecialTokens: false,
      padding: 'do_not_pad',
      truncation: false,
    });

    let inputIds = [...encoded.inputIds];
    const generatedIds: number[] = [];

    // Generation loop
    for (let i = 0; i < maxNewTokens; i++) {
      // Check max length
      if (inputIds.length >= maxLength) break;

      // Run model forward pass
      const nextTokenId = await this.generateNextToken(
        inputIds,
        temperature,
        topK,
        topP,
        repetitionPenalty,
        doSample
      );

      // Check for EOS
      if (nextTokenId === this.eosTokenId) break;

      // Add to sequence
      generatedIds.push(nextTokenId);
      inputIds.push(nextTokenId);

      // Call token callback
      if (options.onToken) {
        const token = this.tokenizer.decode([nextTokenId], true);
        options.onToken(token, nextTokenId);
      }

      // Check stop sequences
      const currentText = this.tokenizer.decode(generatedIds, true);
      let shouldStop = false;
      for (const stopSeq of stopSequences) {
        if (currentText.endsWith(stopSeq)) {
          shouldStop = true;
          break;
        }
      }
      if (shouldStop) break;
    }

    // Decode generated text
    const generatedText = this.tokenizer.decode(generatedIds, true);
    const endTime = performance.now();

    return {
      generatedText,
      fullText: returnFullText ? prompt + generatedText : undefined,
      tokenIds: generatedIds,
      numTokens: generatedIds.length,
      processingTime: endTime - startTime,
    };
  }

  /**
   * Generate next token using the model
   */
  private async generateNextToken(
    inputIds: number[],
    temperature: number,
    topK: number,
    topP: number,
    repetitionPenalty: number,
    doSample: boolean
  ): Promise<number> {
    if (!this.model) {
      throw new Error('Model not loaded');
    }

    // Prepare input tensor
    const inputTensor = new WebInferTensor(
      BigInt64Array.from(inputIds.map(id => BigInt(id))),
      [1, inputIds.length],
      'int64'
    );

    // Create attention mask
    const attentionMask = new WebInferTensor(
      BigInt64Array.from(inputIds.map(() => BigInt(1))),
      [1, inputIds.length],
      'int64'
    );

    // Run inference
    const outputs = await runInference(this.model, [inputTensor, attentionMask]);
    
    if (!outputs || outputs.length === 0) {
      throw new Error('Model returned no outputs');
    }

    // Get logits for last token
    const logits = outputs[0]!;
    const logitsData = logits.toFloat32Array();
    const vocabSize = logits.shape[logits.shape.length - 1] ?? 50257;
    
    // Get logits for the last position
    const lastPositionLogits = new Float32Array(vocabSize);
    const offset = (inputIds.length - 1) * vocabSize;
    
    for (let i = 0; i < vocabSize; i++) {
      lastPositionLogits[i] = logitsData[offset + i] ?? 0;
    }

    // Apply repetition penalty
    if (repetitionPenalty !== 1.0) {
      for (const prevId of inputIds) {
        if (prevId < vocabSize) {
          const score = lastPositionLogits[prevId] ?? 0;
          lastPositionLogits[prevId] = score > 0 
            ? score / repetitionPenalty 
            : score * repetitionPenalty;
        }
      }
    }

    // Apply temperature
    if (temperature !== 1.0) {
      for (let i = 0; i < vocabSize; i++) {
        lastPositionLogits[i] = (lastPositionLogits[i] ?? 0) / temperature;
      }
    }

    // Convert to probabilities
    const logitsTensor = new WebInferTensor(lastPositionLogits, [vocabSize], 'float32');
    const probs = softmax(logitsTensor).toFloat32Array();

    // Sample or greedy
    if (doSample) {
      return this.sample(probs, topK, topP);
    } else {
      return this.greedy(probs);
    }
  }

  /**
   * Greedy decoding (argmax)
   */
  private greedy(probs: Float32Array): number {
    let maxIdx = 0;
    let maxProb = probs[0] ?? 0;
    
    for (let i = 1; i < probs.length; i++) {
      if ((probs[i] ?? 0) > maxProb) {
        maxProb = probs[i] ?? 0;
        maxIdx = i;
      }
    }
    
    return maxIdx;
  }

  /**
   * Sample from probability distribution with top-k/top-p filtering
   */
  private sample(probs: Float32Array, topK: number, topP: number): number {
    // Create sorted indices
    const indices = Array.from({ length: probs.length }, (_, i) => i);
    indices.sort((a, b) => (probs[b] ?? 0) - (probs[a] ?? 0));

    // Apply top-k filtering
    let candidateIndices = indices;
    if (topK > 0 && topK < probs.length) {
      candidateIndices = indices.slice(0, topK);
    }

    // Apply top-p (nucleus) filtering
    if (topP < 1.0) {
      let cumulativeProb = 0;
      const filtered: number[] = [];
      
      for (const idx of candidateIndices) {
        filtered.push(idx);
        cumulativeProb += probs[idx] ?? 0;
        if (cumulativeProb >= topP) break;
      }
      
      candidateIndices = filtered;
    }

    // Renormalize probabilities
    let totalProb = 0;
    for (const idx of candidateIndices) {
      totalProb += probs[idx] ?? 0;
    }

    // Sample
    const r = Math.random() * totalProb;
    let cumulative = 0;
    
    for (const idx of candidateIndices) {
      cumulative += probs[idx] ?? 0;
      if (cumulative >= r) {
        return idx;
      }
    }

    // Fallback
    return candidateIndices[0] ?? 0;
  }

}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create text generation pipeline
 */
export function createTextGenerationPipeline(config?: PipelineConfig): TextGenerationPipeline {
  return new TextGenerationPipeline(config);
}
