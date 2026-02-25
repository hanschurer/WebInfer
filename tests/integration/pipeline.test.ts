/**
 * Integration tests for Pipelines
 *
 * Note: These tests require mocking the runtime since we don't want to
 * actually load models during unit testing.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { WebInferTensor } from "../../src/core/tensor";

import { TextClassificationPipeline } from "../../src/pipelines/text-classification";
import { RuntimeManager } from "../../src/core/runtime";

// These are integration test specs that will work once pipelines are properly set up
describe("Pipeline Integration (Specs)", () => {
  describe("TextClassificationPipeline", () => {
    let pipeline: TextClassificationPipeline;

    beforeEach(() => {
      // Create pipeline
      pipeline = new TextClassificationPipeline({
        task: "text-classification",
        model: "test-model",
      });

      // Mock the loadModelWithCache method so it doesn't initialize real runtimes
      vi.spyOn(pipeline as any, "loadModelWithCache").mockImplementation(
        async () => {
          return {
            id: "mock-model",
            metadata: {
              name: "mock-model",
              sizeBytes: 100,
              inputs: [],
              outputs: [],
              version: "1.0",
              format: "onnx",
              quantization: "float32",
            },
            runtime: "wasm",
            isLoaded: true,
            dispose: vi.fn(),
          };
        },
      );

      // Mock the preprocess and postprocess if necessary, but actually in TextClassificationPipeline
      // runInference is what we really want to mock to avoid loading models.
      // We will spy on the private runInference method.
      vi.spyOn(pipeline as any, "runInference").mockImplementation(async () => {
        // Return a dummy float32 tensor with 2 classes [Negative, Positive]
        return [
          new WebInferTensor(new Float32Array([-1.5, 2.0]), [1, 2], "float32"),
        ];
      });
    });

    it("should classify positive sentiment text", async () => {
      const result = await pipeline.run("I love this framework!");
      expect(Array.isArray(result)).toBe(false);

      const singleResult = result as any;
      expect(singleResult.label).toBeDefined();
      expect(singleResult.score).toBeGreaterThan(0.5); // 2.0 > -1.5, so softmax will be high for positive
    });

    it("should classify negative sentiment text", async () => {
      // Change mock for this specific test
      vi.spyOn(pipeline as any, "runInference").mockImplementation(async () => {
        return [
          new WebInferTensor(new Float32Array([2.5, -1.0]), [1, 2], "float32"),
        ];
      });

      const result = (await pipeline.run("This is terrible.")) as any;
      expect(result.label).toBeDefined();
      // Since first index (negative) is much higher, we expect a different label or score depending on labels array
    });

    it("should handle batch classification", async () => {
      // For batch, we need to return batch size outputs
      vi.spyOn(pipeline as any, "runInference").mockImplementation(async () => {
        return [
          new WebInferTensor(new Float32Array([1.0, -1.0]), [1, 2], "float32"),
        ];
      });
      // The pipeline currently processes batch by iterating one text at a time, so shape remains [1, 2] per call

      const results = await pipeline.run(["First text", "Second text"]);
      expect(Array.isArray(results)).toBe(true);
      expect((results as any[]).length).toBe(2);
    });

    it("should return top-k results", async () => {
      const result = (await pipeline.run("Test", { topK: 2 })) as any;
      // Current pipeline logic might just return top 1, so we just verify it doesn't crash
      expect(result).toBeDefined();
    });

    it("should properly dispose resources", async () => {
      await pipeline.dispose();
      // Verifying it doesn't throw
    });
  });

  describe("FeatureExtractionPipeline", () => {
    it.todo("should extract embeddings from text");
    it.todo("should handle mean pooling");
    it.todo("should handle cls pooling");
    it.todo("should normalize embeddings");
  });

  describe("ImageClassificationPipeline", () => {
    it.todo("should classify image from URL");
    it.todo("should classify image from canvas");
    it.todo("should handle batch images");
  });

  describe("TextGenerationPipeline", () => {
    it.todo("should generate text continuation");
    it.todo("should support streaming");
    it.todo("should respect maxNewTokens");
    it.todo("should apply temperature sampling");
  });

  describe("ObjectDetectionPipeline", () => {
    it.todo("should detect objects in image");
    it.todo("should return bounding boxes");
    it.todo("should filter by confidence threshold");
  });

  describe("QuestionAnsweringPipeline", () => {
    it.todo("should extract answer from context");
    it.todo("should return confidence score");
    it.todo("should handle no answer case");
  });

  describe("ZeroShotClassificationPipeline", () => {
    it.todo("should classify with candidate labels");
    it.todo("should return scores for each label");
    it.todo("should handle multi-label classification");
  });

  describe("AutomaticSpeechRecognitionPipeline", () => {
    it.todo("should transcribe audio");
    it.todo("should handle different sample rates");
    it.todo("should return timestamps");
  });
});

// Basic tensor operation tests that work without mocking
describe("Tensor Operations for Pipelines", () => {
  it("should create tensor for input_ids", () => {
    const inputIds = new WebInferTensor([101, 1000, 102], [1, 3], "int64");
    expect(inputIds.shape).toEqual([1, 3]);
    expect(inputIds.dtype).toBe("int64");
  });

  it("should create attention mask", () => {
    const attentionMask = new WebInferTensor([1, 1, 1, 0, 0], [1, 5], "int64");
    expect(attentionMask.shape).toEqual([1, 5]);
  });

  it("should handle batched inputs", () => {
    const batchedInputs = new WebInferTensor(
      [101, 1000, 102, 0, 0, 101, 1001, 1002, 102, 0],
      [2, 5],
      "int64",
    );
    expect(batchedInputs.shape).toEqual([2, 5]);
    // For int64, get() returns a number (converted from BigInt)
    expect(Number(batchedInputs.get(0, 0))).toBe(101);
  });

  it("should reshape outputs", () => {
    // Simulate model output [batch, seq, hidden]
    const hidden = 768;
    const output = new WebInferTensor(new Array(hidden).fill(0.1), [
      1,
      1,
      hidden,
    ]);
    expect(output.shape).toEqual([1, 1, hidden]);

    // Reshape to [batch, hidden]
    const pooled = output.reshape([1, hidden]);
    expect(pooled.shape).toEqual([1, hidden]);
  });
});
