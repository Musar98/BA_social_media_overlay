import type { InferenceSession, Tensor } from "onnxruntime-web";
import { AIStateType, ONNXState, UIStateType } from "./Types";
import { AIParams } from "../ai/Types";

class AIStateHolder implements AIStateType {
  params: AIParams | undefined = undefined;
}

class UIStateHolder implements UIStateType {
  filterEnabled: boolean = false;
}

class ONNXStateHolder implements ONNXState {
  session: InferenceSession | null = null;
  alphasTensor: Tensor | null = null;
}

export const AIState: AIStateType = new AIStateHolder();
export const UIState: UIStateType = new UIStateHolder();
export const ONNXSessionState: ONNXState = new ONNXStateHolder();
