import { AIStateType, ONNXStateType, UIStateType } from "./Types";

export const ONNXState: ONNXStateType = {
  session: null,
  busy: false,
  alphasTensor: null,
};

//TODO maybe remove default params
export const AIState: AIStateType = {
  params: undefined
};

export const UIState: UIStateType = {
  filterEnabled: false,
};
