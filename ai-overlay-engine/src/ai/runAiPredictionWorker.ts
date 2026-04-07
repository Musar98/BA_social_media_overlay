// ai/runAiPredictionWorker.ts
// Worker-safe AI prediction (no window/state imports)

import { getWorkerONNXState } from "./onnxWorker";

let inputBuffer: Float32Array | null = null;
let inputTensor: any | null = null;
let lastW = 0;
let lastH = 0;

function resizeBuffers(w: number, h: number) {
  const HW = w * h;
  inputBuffer = new Float32Array(3 * HW);

  const ort = (self as any).ort;
  inputTensor = new ort.Tensor("float32", inputBuffer, [1, 3, h, w]);

  lastW = w;
  lastH = h;
}

function prepareInput(img: Uint8ClampedArray) {
  if (!inputBuffer || !inputTensor) {
    throw new Error("Buffers not initialized");
  }

  const HW = lastW * lastH;

  let r = 0,
    g = HW,
    b = 2 * HW;
  const inv255 = 1 / 255;

  for (let i = 0; i < img.length; i += 4) {
    inputBuffer[r++] = img[i] * inv255;
    inputBuffer[g++] = img[i + 1] * inv255;
    inputBuffer[b++] = img[i + 2] * inv255;
  }

  return inputTensor;
}

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

    const params = mapTensorToParams(
      output.transform_params.data as Float32Array,
    );
    return params;
  } catch (err) {
    console.error("AI Prediction in worker failed:", err);
    throw err;
  }
}
