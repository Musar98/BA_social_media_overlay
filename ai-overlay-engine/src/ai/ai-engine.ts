import { AIState, ONNXState } from "../state/state";
import { resizeBuffers, prepareInput } from "./preProcessor";
import { mapTensorToParams } from "./postProcessor";

export async function runAIPrediction(video: HTMLVideoElement) {
  if (!ONNXState.session || ONNXState.busy) return;
  ONNXState.busy = true;

  try {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

    resizeBuffers(canvas.width, canvas.height);
    const input = prepareInput(imgData);

    const output = await ONNXState.session.run({
      images: input,
      alphas: ONNXState.alphasTensor!,
    });

    console.info("AI Prediction finished");

    AIState.params = mapTensorToParams(
      output.transform_params.data as Float32Array,
    );
  } catch (e) {
    console.error("AI Prediction failed:", e);
  } finally {
    ONNXState.busy = false;
  }
}
