import { observeVideos } from "./ui/videoObserver";
import { createButton } from "./ui/button";
import { createOverlayCanvas } from "./ui/canvas";
import { startRenderLoop } from "./renderer/renderLoop";

function modifyVideo(video: HTMLVideoElement) {
  if (video.dataset.filterAttached) return;
  video.dataset.filterAttached = "true";
  video.crossOrigin = "anonymous";

  const canvas = createOverlayCanvas(video);

  startRenderLoop(video, canvas);
}

export function initInstagramFilter() {
  createButton();
  observeVideos(modifyVideo);
}
