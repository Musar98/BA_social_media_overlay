import { AIState } from "../state/state";
import { AIParams } from "./Types";

export function mapTensorToParams(data: Float32Array | any[]): AIParams {
  //TODO clearup if this really holds, always 37 params? => move to global const if true PARAM_AMOUNT = 37
  if (!data || data.length < 37) {
    return AIState.params;
  }
  return {
    sharp: data[0],
    exposure: data[1],
    contrast: data[2],
    saturation: data[3],
    blur: data[4],
    toneCurve: Array.from(data.slice(5, 13)),
    colorCurve: Array.from(data.slice(13, 37)),
  };
}
