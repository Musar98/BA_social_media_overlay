import { describe, it, expect } from "vitest";
import { AIState } from "../src/state/state";
import { AIParams } from "../src/ai/Types";
import { postProcessor } from "../src/ai/PostProcessor";

describe("postProcessor.mapTensorToParams", () => {
  it("throw error when input is null", () => {
    expect(() => {
      postProcessor.mapTensorToParams(null as any);
    }).toThrow("Invalid AI output data");
  });

  it("throws error when input array is too short", () => {
    expect(() => {
      postProcessor.mapTensorToParams(new Float32Array(0));
    }).toThrow("Invalid AI output data");
  });

  it("maps a valid Float32Array of exactly 37 elements correctly", () => {
    const data = new Float32Array(37);
    for (let i = 0; i < 37; i++) data[i] = i;

    const result = postProcessor.mapTensorToParams(data);

    const expected: AIParams = {
      sharp: 0,
      exposure: 1,
      contrast: 2,
      saturation: 3,
      blur: 4,
      toneCurve: Array.from(data.slice(5, 13)),
      colorCurve: Array.from(data.slice(13, 37)),
    };

    expect(result).toEqual(expected);
  });

  it("maps a Float32Array longer than 37 elements, ignoring extra values", () => {
    const data = new Float32Array(50);
    for (let i = 0; i < 50; i++) data[i] = i;

    const result = postProcessor.mapTensorToParams(data);

    const expected: AIParams = {
      sharp: 0,
      exposure: 1,
      contrast: 2,
      saturation: 3,
      blur: 4,
      toneCurve: Array.from(data.slice(5, 13)),
      colorCurve: Array.from(data.slice(13, 37)), // still only 24 elements
    };

    expect(result).toEqual(expected);
  });
});
