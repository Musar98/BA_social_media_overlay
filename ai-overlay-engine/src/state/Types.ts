import type { InferenceSession, Tensor } from "onnxruntime-web";

import { AIParams } from "../ai/Types";

export interface ONNXStateType {
  session: InferenceSession | null;
  busy: boolean;
  alphasTensor: Tensor | null;
}

//TODO extend for other model output
export interface AIStateType {
  params: AIParams;
}

export interface UIStateType {
  filterEnabled: boolean;
}
