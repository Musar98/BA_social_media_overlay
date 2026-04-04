import { AIStateType, ONNXStateType, UIStateType } from "./Types";

export const ONNXState: ONNXStateType = {
  session: null,
  busy: false,
  alphasTensor: null,
};

//TODO maybe remove default params
export const AIState: AIStateType = {
  params: {
    sharp: 0.5,
    exposure: 0.2,
    contrast: 1.1,
    saturation: 1.2,
    blur: 0.0,
    toneCurve: new Array(8).fill(1.0),
    colorCurve: new Array(24).fill(1.0),
  },
};

export const UIState: UIStateType = {
  filterEnabled: false,
};
