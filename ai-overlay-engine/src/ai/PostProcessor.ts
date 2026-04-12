import { AIParams } from "./Types";

class PostProcessor {
  public mapTensorToParams(data: Float32Array | any[]): AIParams {
    if (!data || data.length < 37) {
      throw new Error("Invalid AI output data");
    }
    return {
      sharp: data[0],
      exposure: data[1],
      contrast: data[2],
      saturation: data[3],
      blur: data[4],
      toneCurve: Array.from(data.slice(5, 13)),
      colorCurve: Array.from(data.slice(13, 37)),
    };
  }
}

export const postProcessor = new PostProcessor();
