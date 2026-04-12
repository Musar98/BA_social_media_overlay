import { getONNXState } from "./onnx";
import { preProcessor } from "./PreProcessor";
import { mapTensorToParams } from "./postProcessor";
import { AIParams } from "./Types";

export async function runAiPrediction(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
): Promise<AIParams> {
  const ONNXState = getONNXState();

  if (!ONNXState.session) {
    throw new Error("ONNX session not initialized");
  }

  try {
    const input = preProcessor.prepareInput(width, height, pixels);

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
