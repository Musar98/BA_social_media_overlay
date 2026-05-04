import { AIParams } from "./Types";

const TONE_CURVE_STEPS = 8;
const COLOR_CURVE_STEPS = 8;

class PostProcessor {
  public mapTensorToParams(data: Float32Array | number[]): AIParams {
    if (!data || data.length < 37) {
      throw new Error(`Invalid AI output data: expected at least 37 values, got ${data?.length ?? 0}`);
    }

    const toneCurve = Array.from(data.slice(2, 10));

    // Model/PyTorch flatten order from [B, 3, K, 1]:
    // R0..R7, G0..G7, B0..B7
    //
    // Renderer/WebGL uniform vec3[8] wants:
    // R0,G0,B0, R1,G1,B1, ... R7,G7,B7
    const colorCurve: number[] = [];

    for (let k = 0; k < COLOR_CURVE_STEPS; k++) {
      colorCurve.push(
        data[10 + 0 * COLOR_CURVE_STEPS + k], // Rk
        data[10 + 1 * COLOR_CURVE_STEPS + k], // Gk
        data[10 + 2 * COLOR_CURVE_STEPS + k], // Bk
      );
    }

    return {
      exposure: data[0],
      saturation: data[1],
      toneCurve,
      colorCurve,
      contrast: data[34],
      sharp: data[35],
      blur: data[36],
    };
  }
}

export const postProcessor = new PostProcessor();