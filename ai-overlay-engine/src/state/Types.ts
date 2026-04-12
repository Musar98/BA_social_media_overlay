import { AIParams } from "../ai/Types";
import type { InferenceSession, Tensor } from "onnxruntime-web";

//TODO extend for other model output
export interface AIStateType {
  params: AIParams | undefined;
}

export interface UIStateType {
  filterEnabled: boolean;
}

export interface ONNXState {
  session: InferenceSession | null;
  alphasTensor: Tensor | null;
}
