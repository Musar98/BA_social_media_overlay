import { preProcessor } from "../../src/ai/PreProcessor";
import { describe, expect, it } from "vitest";

class MockTensor {
  type: string;
  data: Float32Array;
  dims: number[];

  constructor(type: string, data: Float32Array, dims: number[]) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}

//needed for mocking onnx runtime tensor
// @ts-ignore
(global as any).self = {
  ort: {
    Tensor: MockTensor,
  },
};

describe("PreProcessor", () => {
  describe("prepareInput", () => {
    it("creates tensor with correct shape", () => {
      const width = 2;
      const height = 3;
      const img = new Uint8ClampedArray(width * height * 4).fill(0);

      const tensor = preProcessor.prepareInput(width, height, img);

      expect(tensor.dims).toEqual([1, 3, height, width]);
    });

    it("normalizes pixel values to [0, 1]", () => {
      const img = new Uint8ClampedArray([
        255,
        128,
        0,
        255, // one pixel
      ]);

      const tensor = preProcessor.prepareInput(1, 1, img);
      const data = tensor.data;

      expect(data[0]).toBeCloseTo(1); // R
      expect(data[1]).toBeCloseTo(128 / 255); // G
      expect(data[2]).toBeCloseTo(0); // B
    });

    it("stores channels in CHW format (R, then G, then B)", () => {
      // 2 pixels
      const img = new Uint8ClampedArray([
        10,
        20,
        30,
        255, // pixel 1
        40,
        50,
        60,
        255, // pixel 2
      ]);

      const tensor = preProcessor.prepareInput(2, 1, img);
      const data = tensor.data;

      const inv255 = 1 / 255;

      // Expect layout:
      // [R1, R2, G1, G2, B1, B2]
      expect(data[0]).toBeCloseTo(10 * inv255);
      expect(data[1]).toBeCloseTo(40 * inv255);

      expect(data[2]).toBeCloseTo(20 * inv255);
      expect(data[3]).toBeCloseTo(50 * inv255);

      expect(data[4]).toBeCloseTo(30 * inv255);
      expect(data[5]).toBeCloseTo(60 * inv255);
    });

    it("handles multi-row images correctly", () => {
      // 2x2 image
      const img = new Uint8ClampedArray([
        // row 1
        1, 2, 3, 255, 4, 5, 6, 255,
        // row 2
        7, 8, 9, 255, 10, 11, 12, 255,
      ]);

      const tensor = preProcessor.prepareInput(2, 2, img);
      const data = tensor.data;
      const inv255 = 1 / 255;

      const HW = 4;

      // R channel
      expect(data[0]).toBeCloseTo(inv255);
      expect(data[1]).toBeCloseTo(4 * inv255);
      expect(data[2]).toBeCloseTo(7 * inv255);
      expect(data[3]).toBeCloseTo(10 * inv255);

      // G channel
      expect(data[HW]).toBeCloseTo(2 * inv255);
      expect(data[HW + 1]).toBeCloseTo(5 * inv255);
      expect(data[HW + 2]).toBeCloseTo(8 * inv255);
      expect(data[HW + 3]).toBeCloseTo(11 * inv255);

      // B channel
      expect(data[2 * HW]).toBeCloseTo(3 * inv255);
      expect(data[2 * HW + 1]).toBeCloseTo(6 * inv255);
      expect(data[2 * HW + 2]).toBeCloseTo(9 * inv255);
      expect(data[2 * HW + 3]).toBeCloseTo(12 * inv255);
    });

    it("ignores alpha channel", () => {
      const img = new Uint8ClampedArray([
        100,
        150,
        200,
        0, // alpha = 0
      ]);

      const tensor = preProcessor.prepareInput(1, 1, img);
      const data = tensor.data;

      expect(data[0]).toBeCloseTo(100 / 255);
      expect(data[1]).toBeCloseTo(150 / 255);
      expect(data[2]).toBeCloseTo(200 / 255);
    });

    it("allocates correct buffer size", () => {
      const width = 4;
      const height = 5;
      const img = new Uint8ClampedArray(width * height * 4).fill(1);

      const tensor = preProcessor.prepareInput(width, height, img);

      expect(tensor.data.length).toBe(3 * width * height);
    });

    it("recreates buffers when size changes", () => {
      const img1 = new Uint8ClampedArray(4).fill(10); // 1x1
      const img2 = new Uint8ClampedArray(16).fill(20); // 2x2

      const t1 = preProcessor.prepareInput(1, 1, img1);
      const t2 = preProcessor.prepareInput(2, 2, img2);

      expect(t1.data.length).toBe(3);
      expect(t2.data.length).toBe(12);
    });

    it("produces deterministic output for same input", () => {
      const img = new Uint8ClampedArray([1, 2, 3, 255, 4, 5, 6, 255]);

      const t1 = preProcessor.prepareInput(2, 1, img);
      const t2 = preProcessor.prepareInput(2, 1, img);

      expect(Array.from(t1.data)).toEqual(Array.from(t2.data));
    });
  });
});
