import type { AIParams } from "./Types";

const IDENTITY_TONE_CURVE = [1, 1, 1, 1, 1, 1, 1, 1];
const IDENTITY_COLOR_CURVE = [
  1, 1, 1,
  1, 1, 1,
  1, 1, 1,
  1, 1, 1,
  1, 1, 1,
  1, 1, 1,
  1, 1, 1,
  1, 1, 1,
];

export function createIdentityAIParams(): AIParams {
  return {
    sharp: 1.0,
    exposure: 0.0,
    contrast: 1.0,
    saturation: 1.0,
    blur: 0.0,
    imageMean: 0.5,
    toneCurve: [...IDENTITY_TONE_CURVE],
    colorCurve: [...IDENTITY_COLOR_CURVE],
  };
}
