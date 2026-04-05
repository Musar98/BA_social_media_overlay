import {
  clipToUV,
  createImageTransformRenderer,
  destroyImageTransformRenderer,
  imageAdjPipeline,
} from "../transformations/transformations";

export function initRenderer(renderer: any, canvas: HTMLCanvasElement) {
  if (renderer) {
    return renderer;
  }

  return createImageTransformRenderer(canvas, clipToUV, imageAdjPipeline, {
    metrics: { enabled: false },
  });
}

export function cleanupRenderer(renderer: any) {
  if (renderer) {
    destroyImageTransformRenderer(renderer);
  }
}
