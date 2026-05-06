export interface AIParams {
  sharp: number;
  exposure: number;
  contrast: number;
  saturation: number;
  blur: number;
  imageMean?: number;
  toneCurve: number[];
  colorCurve: number[];
}
