import { getWorkerONNXState } from "./onnxWorker";
import { prepareInput, resizeBuffers } from "./preProcessor";

function mapTensorToParams(data: Float32Array | any[]): any {
  if (!data || data.length < 37) {
    return undefined;
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

export async function runAIPredictionWorker(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
): Promise<any> {
  const ONNXState = getWorkerONNXState();

  if (!ONNXState.session) {
    throw new Error("ONNX session not initialized");
  }

  try {
    resizeBuffers(width, height);
    const input = prepareInput(pixels);

    const output = await ONNXState.session.run({
      images: input,
      alphas: ONNXState.alphasTensor,
    });

    return mapTensorToParams(output.transform_params.data as Float32Array);
  } catch (err) {
    console.error("AI Prediction in worker failed:", err);
    throw err;
  }
}
