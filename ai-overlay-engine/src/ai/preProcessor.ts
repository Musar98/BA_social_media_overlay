let inputBuffer: Float32Array | null = null;
let inputTensor: any | null = null; // will be a window.ort.Tensor

let lastW = 0;
let lastH = 0;

export function resizeBuffers(w: number, h: number) {
  const HW = w * h;

  inputBuffer = new Float32Array(3 * HW);

  const ort = (window as any).ort;
  inputTensor = new ort.Tensor("float32", inputBuffer, [1, 3, h, w]);

  lastW = w;
  lastH = h;
}

export function prepareInput(img: Uint8ClampedArray) {
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

  return inputTensor; // ready for ONNX session.run()
}
