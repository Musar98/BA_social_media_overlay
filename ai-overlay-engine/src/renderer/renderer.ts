import {
  createImageTransformRenderer,
  destroyImageTransformRenderer,
} from "../transformation/transformations";
import { CLIP_TO_UV } from "../transformation/ClipToUv";
import { IMAGE_ADJUSTMENTS_PIPELINE } from "../transformation/ImageAdjustmentsPipeline";

let renderer: any = null;

export function initRenderer(canvas: HTMLCanvasElement) {
  if (renderer) {
    destroyRenderer();
  }

  renderer = createImageTransformRenderer(
    canvas,
    CLIP_TO_UV,
    IMAGE_ADJUSTMENTS_PIPELINE,
    { metrics: { enabled: false } },
  );

  return renderer;
}

export function destroyRenderer() {
  if (!renderer) {
    return;
  }

  const gl = renderer?.gl;
  if (gl) {
    const ext = gl.getExtension("WEBGL_lose_context");
    ext?.loseContext();
  }

  destroyImageTransformRenderer(renderer);
  renderer = null;
}
