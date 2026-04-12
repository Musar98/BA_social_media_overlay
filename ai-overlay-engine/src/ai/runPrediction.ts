import { getONNXState } from "./onnx";
import { preProcessor } from "./PreProcessor";
import { postProcessor } from "./PostProcessor";
import { AIParams } from "./Types";

export async function runPrediction(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
): Promise<AIParams> {
  const ONNXState = getONNXState();

  if (!ONNXState.alphasTensor) {
    throw new Error("ONNX alphas tensor missing for prediction");
  }

  if (!ONNXState.session) {
    throw new Error("ONNX session not initialized");
  }

  const input = preProcessor.prepareInput(width, height, pixels);

  try {
    const output = await ONNXState.session.run({
      images: input,
      alphas: ONNXState.alphasTensor,
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
