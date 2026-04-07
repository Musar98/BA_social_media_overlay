// ai/onnxWorker.ts
// Worker-safe ONNX initialization (no window/state imports)

interface WorkerONNXState {
  session: any;
  alphasTensor: any;
}

const WorkerONNXState: WorkerONNXState = {
  session: null,
  alphasTensor: null,
};

export async function initORTWorker() {
  if (WorkerONNXState.session) return;

  console.log("Initializing ONNX in worker");

  // Load worker-compatible ONNX runtime
  importScripts("/_onnx/ort.wasm.js"); // ✅ worker-safe

  const ort = (self as any).ort;

  // Set WASM file path
  ort.env.wasm.wasmPaths = "/_onnx/";

  // Fetch your ONNX model
  const response = await fetch("/_onnx/models/parametricmodel.pt.dyn.onnx");
  const modelArrayBuffer = await response.arrayBuffer();

  WorkerONNXState.session = await ort.InferenceSession.create(modelArrayBuffer, {
    executionProviders: ["wasm"],
  });

  WorkerONNXState.alphasTensor = new ort.Tensor("float32", new Float32Array([0.1]), [1]);

  console.log("ONNX initialized in worker");
}

export function getWorkerONNXState() {
  return WorkerONNXState;
}