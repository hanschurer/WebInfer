/**
 * WebInfer.js - Preprocessor
 *
 * Data preprocessing utilities for images, audio, and other data types.
 * Supports HuggingFace preprocessor_config.json format.
 */
import { WebInferTensor } from '../core/tensor.js';
/**
 * Image input types
 */
export type ImageInput = HTMLImageElement | HTMLCanvasElement | ImageBitmap | ImageData | Blob | File | string;
/**
 * Audio input types
 */
export type AudioInput = AudioBuffer | Float32Array | ArrayBuffer | Blob | File | string;
/**
 * Image preprocessing options
 */
export interface ImagePreprocessorOptions {
    /** Target width (or size for square) */
    width?: number;
    /** Target height */
    height?: number;
    /** Single size for square output (sets both width and height) */
    size?: number;
    /** Resize mode */
    resizeMode?: 'stretch' | 'contain' | 'cover' | 'pad' | 'shortest_edge' | 'longest_edge';
    /** Normalization mean */
    mean?: [number, number, number];
    /** Normalization std */
    std?: [number, number, number];
    /** Rescale factor (applied before normalization) */
    rescaleFactor?: number;
    /** Convert to grayscale */
    grayscale?: boolean;
    /** Channel format */
    channelFormat?: 'CHW' | 'HWC';
    /** Output data type */
    dtype?: 'float32' | 'uint8';
    /** Do resize */
    doResize?: boolean;
    /** Do rescale */
    doRescale?: boolean;
    /** Do normalize */
    doNormalize?: boolean;
    /** Do center crop */
    doCenterCrop?: boolean;
    /** Center crop size */
    cropSize?: number | {
        width: number;
        height: number;
    };
    /** Padding color for 'pad' mode (RGB 0-255) */
    paddingColor?: [number, number, number];
}
/**
 * ImagePreprocessor - Process images for model input
 *
 * Supports HuggingFace preprocessor_config.json format.
 */
export declare class ImagePreprocessor {
    private readonly options;
    private canvas;
    private ctx;
    constructor(options?: ImagePreprocessorOptions);
    /**
     * Load from HuggingFace preprocessor_config.json
     */
    static fromConfig(config: Record<string, unknown>): ImagePreprocessor;
    /**
     * Load from HuggingFace Hub
     */
    static fromUrl(url: string): Promise<ImagePreprocessor>;
    /**
     * Load from HuggingFace Hub by model ID
     */
    static fromHuggingFace(modelId: string, options?: {
        revision?: string;
    }): Promise<ImagePreprocessor>;
    /**
     * Initialize canvas (lazy)
     */
    private ensureCanvas;
    /**
     * Process an image
     */
    process(input: ImageInput): Promise<WebInferTensor>;
    /**
     * Process multiple images (batch)
     */
    processBatch(inputs: ImageInput[]): Promise<WebInferTensor>;
    /**
     * Load image from URL or base64
     */
    private loadFromUrl;
    /**
     * Load image from Blob/File
     */
    private loadFromBlob;
    /**
     * Center crop image
     */
    private centerCrop;
    /**
     * Convert image element to ImageData
     */
    private toImageData;
    /**
     * Resize image data
     */
    private resize;
    /**
     * Convert ImageData to tensor
     */
    private toTensor;
    /**
     * Get current options
     */
    getOptions(): ImagePreprocessorOptions;
}
/**
 * Audio preprocessing options
 */
export interface AudioPreprocessorOptions {
    /** Target sample rate */
    sampleRate?: number;
    /** Number of mel bins */
    nMels?: number;
    /** FFT size */
    nFft?: number;
    /** Hop length */
    hopLength?: number;
    /** Whether to normalize */
    normalize?: boolean;
    /** Maximum duration in seconds */
    maxDuration?: number;
}
/**
 * AudioPreprocessor - Process audio for model input
 *
 * Supports Whisper and other audio model preprocessing.
 */
export declare class AudioPreprocessor {
    private readonly options;
    private audioContext;
    constructor(options?: AudioPreprocessorOptions);
    /**
     * Load from HuggingFace feature_extractor config
     */
    static fromConfig(config: Record<string, unknown>): AudioPreprocessor;
    /**
     * Load from HuggingFace Hub
     */
    static fromHuggingFace(modelId: string, options?: {
        revision?: string;
    }): Promise<AudioPreprocessor>;
    /**
     * Initialize audio context (lazy)
     */
    private ensureAudioContext;
    /**
     * Process audio data
     */
    process(input: AudioInput): Promise<WebInferTensor>;
    /**
     * Process raw waveform (for models that don't need mel spectrogram)
     */
    processRaw(input: AudioInput): Promise<WebInferTensor>;
    /**
     * Load audio from URL
     */
    private loadFromUrl;
    /**
     * Load audio from Blob/File
     */
    private loadFromBlob;
    /**
     * Decode audio data
     */
    private decodeAudioData;
    /**
     * Convert AudioBuffer to Float32Array
     */
    private audioBufferToFloat32;
    /**
     * Normalize audio
     */
    private normalizeAudio;
    /**
     * Compute mel spectrogram (simplified implementation)
     */
    private computeMelSpectrogram;
    /**
     * Dispose resources
     */
    dispose(): void;
}
/**
 * Text preprocessing options
 */
export interface TextPreprocessorOptions {
    /** Convert to lowercase */
    lowercase?: boolean;
    /** Remove punctuation */
    removePunctuation?: boolean;
    /** Remove extra whitespace */
    normalizeWhitespace?: boolean;
    /** Maximum length in characters */
    maxLength?: number;
}
/**
 * Preprocess text
 */
export declare function preprocessText(text: string, options?: TextPreprocessorOptions): string;
/**
 * Create image preprocessor with common presets
 */
export declare function createImagePreprocessor(preset?: 'imagenet' | 'clip' | 'vit' | 'custom', options?: ImagePreprocessorOptions): ImagePreprocessor;
/**
 * Create audio preprocessor with common presets
 */
export declare function createAudioPreprocessor(preset?: 'whisper' | 'wav2vec' | 'custom', options?: AudioPreprocessorOptions): AudioPreprocessor;
//# sourceMappingURL=preprocessor.d.ts.map