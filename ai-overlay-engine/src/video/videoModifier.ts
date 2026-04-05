import { createOverlayCanvas } from "../ui/canvas";
import { startRenderLoop } from "../renderer/renderLoop";

export function modifyVideo(video: HTMLVideoElement) {
  if (video.dataset.filterAttached) {
    return;
  }
  video.dataset.filterAttached = "true";
  video.crossOrigin = "anonymous";

  const canvas = createOverlayCanvas(video);

  startRenderLoop(video, canvas);
}
