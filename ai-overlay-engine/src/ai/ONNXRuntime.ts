import { ONNXSessionState } from "../state/state";

class ONNXRuntime {
  async init(): Promise<void> {
    if (ONNXSessionState.session) {
      return;
    }
    console.log("Initializing ONNX in worker");
    importScripts("/_onnx/ort.wasm.js");
    const ort = (self as any).ort;
    ort.env.wasm.wasmPaths = "/_onnx/";

    const response = await fetch("/_onnx/models/parametricmodel.pt.dyn.onnx");
    const modelArrayBuffer = await response.arrayBuffer();

    ONNXSessionState.session = await ort.InferenceSession.create(
      modelArrayBuffer,
      {
        executionProviders: ["wasm"],
      },
    );

    //TODO setzen über arousal/valence
    ONNXSessionState.alphasTensor = new ort.Tensor(
      "float32",
      new Float32Array([-0.05]),
      [1],
    );

    console.log("ONNX initialized in worker");
  }
}

export const onnxRuntime = new ONNXRuntime();
