import { ONNXState } from "../state/state";

export async function initORT() {
  if (ONNXState.session) {
    return ONNXState.session;
  }

  await new Promise<void>((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "/_onnx/ort.min.js";
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Failed to load ONNX runtime"));
    document.head.appendChild(s);
  });

  const ort = (window as any).ort;

  ort.env.wasm.wasmPaths = "/_onnx/";

  const response = await fetch("/_onnx/models/parametricmodel.pt.dyn.onnx");
  const modelArrayBuffer = await response.arrayBuffer();

  //TODO try to fix using webgpu and reuse commented out code again
  // const providers = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];

  const providers = ["wasm"];

  ONNXState.session = await ort.InferenceSession.create(modelArrayBuffer, {
    executionProviders: providers,
  });

  ONNXState.alphasTensor = new ort.Tensor(
    "float32",
    new Float32Array([0.1]),
    [1],
  );

  console.info("ONNX Runtime loaded");

  return ONNXState.session;
}
