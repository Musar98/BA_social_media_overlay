import { getWorkerONNXState } from "./onnxWorker";
import { prepareInput, resizeBuffers } from "./preProcessor";
import { mapTensorToParams } from "./postProcessor";

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
