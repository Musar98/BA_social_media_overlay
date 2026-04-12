import { AIStateType, ONNXState, UIStateType } from "./Types";

export const AIState: AIStateType = {
  params: undefined
};

export const UIState: UIStateType = {
  filterEnabled: false,
};

export const ONNXSessionState: ONNXState = {
  session: null,
  alphasTensor: null,
};