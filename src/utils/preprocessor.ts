/**
 * WebInfer.js - Preprocessor
 * 
 * Data preprocessing utilities for images, audio, and other data types.
 * Supports HuggingFace preprocessor_config.json format.
 */

import { WebInferTensor } from '../core/tensor.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Image input types
 */
export type ImageInput = 
  | HTMLImageElement 
  | HTMLCanvasElement 
  | ImageBitmap 
  | ImageData 
  | Blob 
  | File 
  | string;

/**
 * Audio input types
 */
export type AudioInput = 
  | AudioBuffer 
  | Float32Array 
  | ArrayBuffer 
  | Blob 
  | File 
  | string;

// ============================================================================
// Image Preprocessing
// ============================================================================

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
  cropSize?: number | { width: number; height: number };
  /** Padding color for 'pad' mode (RGB 0-255) */
  paddingColor?: [number, number, number];
}

/**
 * Default image preprocessing options (ImageNet style)
 */
const DEFAULT_IMAGE_OPTIONS: ImagePreprocessorOptions = {
  width: 224,
  height: 224,
  resizeMode: 'cover',
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  rescaleFactor: 1 / 255,
  grayscale: false,
  channelFormat: 'CHW',
  dtype: 'float32',
  doResize: true,
  doRescale: true,
  doNormalize: true,
  doCenterCrop: false,
  paddingColor: [0, 0, 0],
};

/**
 * ImagePreprocessor - Process images for model input
 * 
 * Supports HuggingFace preprocessor_config.json format.
 */
export class ImagePreprocessor {
  private readonly options: Required<ImagePreprocessorOptions>;
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;

  constructor(options: ImagePreprocessorOptions = {}) {
    // Handle size option
    const size = options.size;
    const width = options.width ?? size ?? DEFAULT_IMAGE_OPTIONS.width!;
    const height = options.height ?? size ?? DEFAULT_IMAGE_OPTIONS.height!;
    
    this.options = {
      ...DEFAULT_IMAGE_OPTIONS,
      ...options,
      width,
      height,
      size: size ?? width,
      cropSize: options.cropSize ?? options.size ?? width,
    } as Required<ImagePreprocessorOptions>;
  }

  /**
   * Load from HuggingFace preprocessor_config.json
   */
  static fromConfig(config: Record<string, unknown>): ImagePreprocessor {
    const options: ImagePreprocessorOptions = {};
    
    // Map HuggingFace config to our options
    const size = config['size'];
    if (size !== undefined) {
      if (typeof size === 'number') {
        options.size = size;
      } else if (typeof size === 'object' && size !== null) {
        const sizeObj = size as { width?: number; height?: number; shortest_edge?: number };
        options.width = sizeObj.width ?? sizeObj.shortest_edge;
        options.height = sizeObj.height ?? sizeObj.shortest_edge;
      }
    }
    
    const cropSize = config['crop_size'];
    if (cropSize !== undefined) {
      if (typeof cropSize === 'number') {
        options.cropSize = cropSize;
      } else if (typeof cropSize === 'object' && cropSize !== null) {
        const cropObj = cropSize as { width?: number; height?: number };
        options.cropSize = { width: cropObj.width ?? 224, height: cropObj.height ?? 224 };
      }
    }
    
    const imageMean = config['image_mean'];
    if (Array.isArray(imageMean)) {
      options.mean = imageMean as [number, number, number];
    }
    
    const imageStd = config['image_std'];
    if (Array.isArray(imageStd)) {
      options.std = imageStd as [number, number, number];
    }
    
    const rescaleFactor = config['rescale_factor'];
    if (typeof rescaleFactor === 'number') {
      options.rescaleFactor = rescaleFactor;
    }
    
    const doResize = config['do_resize'];
    if (typeof doResize === 'boolean') {
      options.doResize = doResize;
    }
    
    const doRescale = config['do_rescale'];
    if (typeof doRescale === 'boolean') {
      options.doRescale = doRescale;
    }
    
    const doNormalize = config['do_normalize'];
    if (typeof doNormalize === 'boolean') {
      options.doNormalize = doNormalize;
    }
    
    const doCenterCrop = config['do_center_crop'];
    if (typeof doCenterCrop === 'boolean') {
      options.doCenterCrop = doCenterCrop;
    }
    
    if (config['resample'] !== undefined) {
      // Map HuggingFace resample to our resize mode
      options.resizeMode = 'cover';
    }
    
    return new ImagePreprocessor(options);
  }

  /**
   * Load from HuggingFace Hub
   */
  static async fromUrl(url: string): Promise<ImagePreprocessor> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load preprocessor config from ${url}`);
    }
    const config = await response.json() as Record<string, unknown>;
    return ImagePreprocessor.fromConfig(config);
  }

  /**
   * Load from HuggingFace Hub by model ID
   */
  static async fromHuggingFace(
    modelId: string,
    options?: { revision?: string }
  ): Promise<ImagePreprocessor> {
    const revision = options?.revision ?? 'main';
    const url = `https://huggingface.co/${modelId}/resolve/${revision}/preprocessor_config.json`;
    return ImagePreprocessor.fromUrl(url);
  }

  /**
   * Initialize canvas (lazy)
   */
  private ensureCanvas(): void {
    if (!this.canvas) {
      if (typeof document !== 'undefined') {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
      } else {
        throw new Error('ImagePreprocessor requires a browser environment');
      }
    }
  }

  /**
   * Process an image
   */
  async process(input: ImageInput): Promise<WebInferTensor> {
    let imageData: ImageData;

    if (typeof input === 'string') {
      // Load from URL or base64
      imageData = await this.loadFromUrl(input);
    } else if (input instanceof Blob || input instanceof File) {
      imageData = await this.loadFromBlob(input);
    } else if (input instanceof ImageData) {
      imageData = input;
    } else {
      // HTMLImageElement, HTMLCanvasElement, ImageBitmap
      imageData = this.toImageData(input);
    }

    // Apply preprocessing pipeline
    let processed = imageData;

    // 1. Resize
    if (this.options.doResize) {
      processed = this.resize(processed);
    }

    // 2. Center crop
    if (this.options.doCenterCrop) {
      processed = this.centerCrop(processed);
    }

    // 3. Convert to tensor (with rescale and normalize)
    return this.toTensor(processed);
  }

  /**
   * Process multiple images (batch)
   */
  async processBatch(inputs: ImageInput[]): Promise<WebInferTensor> {
    const tensors = await Promise.all(inputs.map(input => this.process(input)));
    
    // Stack tensors into batch
    const batchSize = tensors.length;
    const firstTensor = tensors[0];
    if (!firstTensor) {
      return new WebInferTensor(new Float32Array(0), [0], 'float32');
    }
    
    const channels = firstTensor.shape[0] ?? 3;
    const height = firstTensor.shape[1] ?? this.options.height;
    const width = firstTensor.shape[2] ?? this.options.width;
    
    const batchData = new Float32Array(batchSize * channels * height * width);
    
    for (let i = 0; i < tensors.length; i++) {
      const t = tensors[i];
      if (t) {
        batchData.set(t.toFloat32Array(), i * channels * height * width);
      }
    }

    return new WebInferTensor(
      batchData,
      [batchSize, channels, height, width],
      'float32'
    );
  }

  /**
   * Load image from URL or base64
   */
  private async loadFromUrl(url: string): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      img.onload = () => {
        resolve(this.toImageData(img));
      };
      
      img.onerror = () => {
        reject(new Error(`Failed to load image from ${url}`));
      };
      
      img.src = url;
    });
  }

  /**
   * Load image from Blob/File
   */
  private async loadFromBlob(blob: Blob): Promise<ImageData> {
    const url = URL.createObjectURL(blob);
    try {
      return await this.loadFromUrl(url);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  /**
   * Center crop image
   */
  private centerCrop(imageData: ImageData): ImageData {
    const cropSize = this.options.cropSize;
    let cropWidth: number;
    let cropHeight: number;
    
    if (typeof cropSize === 'number') {
      cropWidth = cropSize;
      cropHeight = cropSize;
    } else {
      cropWidth = cropSize.width;
      cropHeight = cropSize.height;
    }
    
    const srcX = Math.max(0, Math.floor((imageData.width - cropWidth) / 2));
    const srcY = Math.max(0, Math.floor((imageData.height - cropHeight) / 2));
    
    this.ensureCanvas();
    
    // Draw source image
    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = imageData.width;
    srcCanvas.height = imageData.height;
    const srcCtx = srcCanvas.getContext('2d')!;
    srcCtx.putImageData(imageData, 0, 0);
    
    // Crop
    this.canvas!.width = cropWidth;
    this.canvas!.height = cropHeight;
    this.ctx!.drawImage(srcCanvas, srcX, srcY, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);
    
    return this.ctx!.getImageData(0, 0, cropWidth, cropHeight);
  }

  /**
   * Convert image element to ImageData
   */
  private toImageData(
    source: HTMLImageElement | HTMLCanvasElement | ImageBitmap
  ): ImageData {
    this.ensureCanvas();
    
    const { width, height } = source;
    this.canvas!.width = width;
    this.canvas!.height = height;
    
    this.ctx!.drawImage(source, 0, 0);
    return this.ctx!.getImageData(0, 0, width, height);
  }

  /**
   * Resize image data
   */
  private resize(imageData: ImageData): ImageData {
    const { width, height, resizeMode } = this.options;
    
    this.ensureCanvas();
    
    // Calculate resize dimensions
    let srcX = 0, srcY = 0, srcW = imageData.width, srcH = imageData.height;
    let dstX = 0, dstY = 0, dstW = width, dstH = height;

    if (resizeMode === 'contain') {
      const scale = Math.min(width / imageData.width, height / imageData.height);
      dstW = Math.round(imageData.width * scale);
      dstH = Math.round(imageData.height * scale);
      dstX = Math.round((width - dstW) / 2);
      dstY = Math.round((height - dstH) / 2);
    } else if (resizeMode === 'cover') {
      const scale = Math.max(width / imageData.width, height / imageData.height);
      srcW = Math.round(width / scale);
      srcH = Math.round(height / scale);
      srcX = Math.round((imageData.width - srcW) / 2);
      srcY = Math.round((imageData.height - srcH) / 2);
    }

    // Create temp canvas for source
    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = imageData.width;
    srcCanvas.height = imageData.height;
    const srcCtx = srcCanvas.getContext('2d')!;
    srcCtx.putImageData(imageData, 0, 0);

    // Draw to output canvas
    this.canvas!.width = width;
    this.canvas!.height = height;
    
    // Fill with black for padding modes
    if (resizeMode === 'contain' || resizeMode === 'pad') {
      this.ctx!.fillStyle = 'black';
      this.ctx!.fillRect(0, 0, width, height);
    }
    
    this.ctx!.drawImage(srcCanvas, srcX, srcY, srcW, srcH, dstX, dstY, dstW, dstH);
    
    return this.ctx!.getImageData(0, 0, width, height);
  }

  /**
   * Convert ImageData to tensor
   */
  private toTensor(imageData: ImageData): WebInferTensor {
    const { 
      mean, std, grayscale, channelFormat, dtype,
      doRescale, rescaleFactor, doNormalize
    } = this.options;
    
    const height = imageData.height;
    const width = imageData.width;
    const channels = grayscale ? 1 : 3;
    
    const data = new Float32Array(channels * height * width);
    const pixels = imageData.data;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelIdx = (y * width + x) * 4;
        
        if (grayscale) {
          // Convert to grayscale
          let gray = (
            0.299 * (pixels[pixelIdx] ?? 0) +
            0.587 * (pixels[pixelIdx + 1] ?? 0) +
            0.114 * (pixels[pixelIdx + 2] ?? 0)
          );
          
          if (doRescale) {
            gray *= rescaleFactor;
          }
          
          if (doNormalize) {
            gray = (gray - (mean[0] ?? 0)) / (std[0] ?? 1);
          }
          
          const idx = y * width + x;
          data[idx] = gray;
        } else if (channelFormat === 'CHW') {
          // Channel-first format (used by most PyTorch models)
          for (let c = 0; c < 3; c++) {
            let value = pixels[pixelIdx + c] ?? 0;
            
            if (doRescale) {
              value *= rescaleFactor;
            }
            
            if (doNormalize) {
              value = (value - (mean[c] ?? 0)) / (std[c] ?? 1);
            }
            
            const idx = c * height * width + y * width + x;
            data[idx] = value;
          }
        } else {
          // HWC format (used by TensorFlow models)
          for (let c = 0; c < 3; c++) {
            let value = pixels[pixelIdx + c] ?? 0;
            
            if (doRescale) {
              value *= rescaleFactor;
            }
            
            if (doNormalize) {
              value = (value - (mean[c] ?? 0)) / (std[c] ?? 1);
            }
            
            const idx = y * width * 3 + x * 3 + c;
            data[idx] = value;
          }
        }
      }
    }

    const shape = channelFormat === 'CHW'
      ? [channels, height, width]
      : [height, width, channels];

    return new WebInferTensor(data, shape, dtype);
  }

  /**
   * Get current options
   */
  getOptions(): ImagePreprocessorOptions {
    return { ...this.options };
  }
}

// ============================================================================
// Audio Preprocessing
// ============================================================================

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
 * Default audio options
 */
const DEFAULT_AUDIO_OPTIONS: Required<AudioPreprocessorOptions> = {
  sampleRate: 16000,
  nMels: 80,
  nFft: 400,
  hopLength: 160,
  normalize: true,
  maxDuration: 30,
};

/**
 * AudioPreprocessor - Process audio for model input
 * 
 * Supports Whisper and other audio model preprocessing.
 */
export class AudioPreprocessor {
  private readonly options: Required<AudioPreprocessorOptions>;
  private audioContext: AudioContext | null = null;

  constructor(options: AudioPreprocessorOptions = {}) {
    this.options = { ...DEFAULT_AUDIO_OPTIONS, ...options };
  }

  /**
   * Load from HuggingFace feature_extractor config
   */
  static fromConfig(config: Record<string, unknown>): AudioPreprocessor {
    const options: AudioPreprocessorOptions = {};
    
    const samplingRate = config['sampling_rate'];
    if (typeof samplingRate === 'number') {
      options.sampleRate = samplingRate;
    }
    
    const featureSize = config['feature_size'];
    if (typeof featureSize === 'number') {
      options.nMels = featureSize;
    }
    
    const nFft = config['n_fft'];
    if (typeof nFft === 'number') {
      options.nFft = nFft;
    }
    
    const hopLength = config['hop_length'];
    if (typeof hopLength === 'number') {
      options.hopLength = hopLength;
    }
    
    return new AudioPreprocessor(options);
  }

  /**
   * Load from HuggingFace Hub
   */
  static async fromHuggingFace(
    modelId: string,
    options?: { revision?: string }
  ): Promise<AudioPreprocessor> {
    const revision = options?.revision ?? 'main';
    const url = `https://huggingface.co/${modelId}/resolve/${revision}/preprocessor_config.json`;
    
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load audio config from ${url}`);
    }
    const config = await response.json() as Record<string, unknown>;
    return AudioPreprocessor.fromConfig(config);
  }

  /**
   * Initialize audio context (lazy)
   */
  private ensureAudioContext(): void {
    if (!this.audioContext) {
      if (typeof AudioContext !== 'undefined') {
        this.audioContext = new AudioContext({ sampleRate: this.options.sampleRate });
      } else {
        throw new Error('AudioPreprocessor requires Web Audio API support');
      }
    }
  }

  /**
   * Process audio data
   */
  async process(input: AudioInput): Promise<WebInferTensor> {
    let audioData: Float32Array;

    if (typeof input === 'string') {
      // Load from URL
      audioData = await this.loadFromUrl(input);
    } else if (input instanceof Blob || input instanceof File) {
      // Load from Blob/File
      audioData = await this.loadFromBlob(input);
    } else if (input instanceof AudioBuffer) {
      audioData = this.audioBufferToFloat32(input);
    } else if (input instanceof Float32Array) {
      audioData = input;
    } else {
      // ArrayBuffer - decode
      audioData = await this.decodeAudioData(input);
    }

    // Resample if needed
    // For now, assume input is at target sample rate

    // Normalize
    if (this.options.normalize) {
      audioData = this.normalizeAudio(audioData);
    }

    // Truncate if needed
    const maxSamples = this.options.maxDuration * this.options.sampleRate;
    if (audioData.length > maxSamples) {
      audioData = audioData.slice(0, maxSamples);
    }

    // Compute mel spectrogram (simplified)
    const melSpec = this.computeMelSpectrogram(audioData);

    return melSpec;
  }

  /**
   * Process raw waveform (for models that don't need mel spectrogram)
   */
  async processRaw(input: AudioInput): Promise<WebInferTensor> {
    let audioData: Float32Array;

    if (typeof input === 'string') {
      audioData = await this.loadFromUrl(input);
    } else if (input instanceof Blob || input instanceof File) {
      audioData = await this.loadFromBlob(input);
    } else if (input instanceof AudioBuffer) {
      audioData = this.audioBufferToFloat32(input);
    } else if (input instanceof Float32Array) {
      audioData = input;
    } else {
      audioData = await this.decodeAudioData(input);
    }

    // Normalize
    if (this.options.normalize) {
      audioData = this.normalizeAudio(audioData);
    }

    // Truncate/pad
    const maxSamples = this.options.maxDuration * this.options.sampleRate;
    if (audioData.length > maxSamples) {
      audioData = audioData.slice(0, maxSamples);
    }

    return new WebInferTensor(audioData, [1, audioData.length], 'float32');
  }

  /**
   * Load audio from URL
   */
  private async loadFromUrl(url: string): Promise<Float32Array> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load audio from ${url}`);
    }
    
    const arrayBuffer = await response.arrayBuffer();
    return this.decodeAudioData(arrayBuffer);
  }

  /**
   * Load audio from Blob/File
   */
  private async loadFromBlob(blob: Blob): Promise<Float32Array> {
    const arrayBuffer = await blob.arrayBuffer();
    return this.decodeAudioData(arrayBuffer);
  }

  /**
   * Decode audio data
   */
  private async decodeAudioData(data: ArrayBuffer): Promise<Float32Array> {
    this.ensureAudioContext();
    const audioBuffer = await this.audioContext!.decodeAudioData(data.slice(0)); // Clone to avoid detached buffer
    return this.audioBufferToFloat32(audioBuffer);
  }

  /**
   * Convert AudioBuffer to Float32Array
   */
  private audioBufferToFloat32(buffer: AudioBuffer): Float32Array {
    // Get first channel
    const channelData = buffer.getChannelData(0);
    return new Float32Array(channelData);
  }

  /**
   * Normalize audio
   */
  private normalizeAudio(data: Float32Array): Float32Array {
    let max = 0;
    for (let i = 0; i < data.length; i++) {
      const abs = Math.abs(data[i] ?? 0);
      if (abs > max) max = abs;
    }

    if (max > 0) {
      const result = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) {
        result[i] = (data[i] ?? 0) / max;
      }
      return result;
    }

    return data;
  }

  /**
   * Compute mel spectrogram (simplified implementation)
   */
  private computeMelSpectrogram(audio: Float32Array): WebInferTensor {
    const { nMels, nFft, hopLength } = this.options;
    
    // Calculate number of frames
    const numFrames = Math.floor((audio.length - nFft) / hopLength) + 1;
    
    if (numFrames <= 0) {
      // Return empty spectrogram for very short audio
      return new WebInferTensor(new Float32Array(nMels), [1, nMels], 'float32');
    }

    const melSpec = new Float32Array(numFrames * nMels);

    // Simplified mel spectrogram computation
    // In production, use proper FFT and mel filterbank
    for (let frame = 0; frame < numFrames; frame++) {
      const start = frame * hopLength;
      
      // Compute frame energy (simplified - not real FFT)
      for (let mel = 0; mel < nMels; mel++) {
        let energy = 0;
        const freqStart = Math.floor((mel / nMels) * (nFft / 2));
        const freqEnd = Math.floor(((mel + 1) / nMels) * (nFft / 2));
        
        for (let i = freqStart; i < Math.min(freqEnd, nFft); i++) {
          const sample = audio[start + i] ?? 0;
          energy += sample * sample;
        }
        
        // Convert to log scale
        melSpec[frame * nMels + mel] = Math.log(energy + 1e-10);
      }
    }

    return new WebInferTensor(melSpec, [numFrames, nMels], 'float32');
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}

// ============================================================================
// Text Preprocessing
// ============================================================================

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
export function preprocessText(
  text: string,
  options: TextPreprocessorOptions = {}
): string {
  const {
    lowercase = true,
    removePunctuation = false,
    normalizeWhitespace = true,
    maxLength,
  } = options;

  let result = text;

  if (lowercase) {
    result = result.toLowerCase();
  }

  if (removePunctuation) {
    result = result.replace(/[^\w\s]/g, '');
  }

  if (normalizeWhitespace) {
    result = result.replace(/\s+/g, ' ').trim();
  }

  if (maxLength && result.length > maxLength) {
    result = result.slice(0, maxLength);
  }

  return result;
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create image preprocessor with common presets
 */
export function createImagePreprocessor(
  preset: 'imagenet' | 'clip' | 'vit' | 'custom' = 'imagenet',
  options: ImagePreprocessorOptions = {}
): ImagePreprocessor {
  const presets: Record<string, ImagePreprocessorOptions> = {
    imagenet: {
      width: 224,
      height: 224,
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
    },
    clip: {
      width: 224,
      height: 224,
      mean: [0.48145466, 0.4578275, 0.40821073],
      std: [0.26862954, 0.26130258, 0.27577711],
    },
    vit: {
      width: 224,
      height: 224,
      mean: [0.5, 0.5, 0.5],
      std: [0.5, 0.5, 0.5],
    },
    custom: {},
  };

  return new ImagePreprocessor({ ...presets[preset], ...options });
}

/**
 * Create audio preprocessor with common presets
 */
export function createAudioPreprocessor(
  preset: 'whisper' | 'wav2vec' | 'custom' = 'whisper',
  options: AudioPreprocessorOptions = {}
): AudioPreprocessor {
  const presets: Record<string, AudioPreprocessorOptions> = {
    whisper: {
      sampleRate: 16000,
      nMels: 80,
      nFft: 400,
      hopLength: 160,
    },
    wav2vec: {
      sampleRate: 16000,
      normalize: true,
    },
    custom: {},
  };

  return new AudioPreprocessor({ ...presets[preset], ...options });
}
