
import { AIParams } from "../ai/Types";

//TODO extend for other model output
export interface AIStateType {
  params: AIParams | undefined;
}

export interface UIStateType {
  filterEnabled: boolean;
}
