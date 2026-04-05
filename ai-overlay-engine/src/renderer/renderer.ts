import {
  createImageTransformRenderer,
  destroyImageTransformRenderer,
} from "../transformations/transformations";
import { CLIP_TO_UV } from "../transformations/ClipToUv";
import { IMAGE_ADJUSTMENTS_PIPELINE } from "../transformations/ImageAdjustmentsPipeline";

export function initRenderer(renderer: any, canvas: HTMLCanvasElement) {
  if (renderer) {
    return renderer;
  }

  return createImageTransformRenderer(
    canvas,
    CLIP_TO_UV,
    IMAGE_ADJUSTMENTS_PIPELINE,
    {
      metrics: { enabled: false },
    },
  );
}

export function cleanupRenderer(renderer: any) {
  if (renderer) {
    destroyImageTransformRenderer(renderer);
  }
}
