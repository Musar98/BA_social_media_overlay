import { AIState, ONNXState } from "../state/state";
import { resizeBuffers, prepareInput } from "./preProcessor";
import { mapTensorToParams } from "./postProcessor";

export async function runAIPrediction(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
) {
  if (!ONNXState.session || ONNXState.busy) {
    return;
  }

  ONNXState.busy = true;

  try {
    resizeBuffers(width, height);
    const input = prepareInput(pixels);

    const output = await ONNXState.session.run({
      images: input,
      alphas: ONNXState.alphasTensor!,
    });

    console.info("AI Prediction finished");

    const params = mapTensorToParams(
      output.transform_params.data as Float32Array,
    );
    AIState.params = params;
    return params;
  } catch (e) {
    console.error("AI Prediction failed:", e);
    throw e;
  } finally {
    ONNXState.busy = false;
  }
}
