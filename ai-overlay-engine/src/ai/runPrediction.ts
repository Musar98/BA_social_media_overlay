import { preProcessor } from "./PreProcessor";
import { postProcessor } from "./PostProcessor";
import { AIParams } from "./Types";
import { ONNXSessionState } from "../state/state";

export async function runPrediction(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
): Promise<AIParams> {
  if (!ONNXSessionState.alphasTensor) {
    throw new Error("ONNX alphas tensor missing for prediction");
  }

  if (!ONNXSessionState.session) {
    throw new Error("ONNX session not initialized");
  }

  const input = preProcessor.prepareInput(width, height, pixels);

  try {
    const output = await ONNXSessionState.session.run({
      images: input,
      alphas: ONNXSessionState.alphasTensor,
    });

    //TODO maybe extract mapTensorToParams this is
    // sep concern of mapping output, not running the prediction itself
    return postProcessor.mapTensorToParams(
      output.transform_params.data as Float32Array,
    );
  } catch (err) {
    console.error("AI Prediction in worker failed:", err);
    throw err;
  }
}
