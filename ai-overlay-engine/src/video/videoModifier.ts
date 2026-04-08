import { createOverlayCanvas } from "../ui/canvas";
import { startRenderLoop, stopRenderLoop } from "../renderer/renderLoop";

let currentVideo: HTMLVideoElement | null = null;
let currentCanvas: HTMLCanvasElement | null = null;

export function modifyVideo(video: HTMLVideoElement) {
  if (video === currentVideo) return;

  document.querySelectorAll("video").forEach((v) => {
    if (v !== video) {
      v.dataset.filterAttached = "true";
    }
  });

  video.dataset.filterAttached = "true";
  video.crossOrigin = "anonymous";

  currentVideo = video;
  currentCanvas = createOverlayCanvas(video);

  startRenderLoop(video, currentCanvas);
}

export function cleanupCurrentVideo() {
  if (currentVideo) {
    stopRenderLoop();
    currentVideo = null;
  }

  if (currentCanvas) {
    currentCanvas.remove();
    currentCanvas = null;
  }
}
