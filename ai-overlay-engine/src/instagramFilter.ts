import { runAIPrediction } from "./ai/ai-engine";
import { observeVideos } from "./ui/videoObserver";
import {
  clipToUV,
  createImageTransformRenderer,
  destroyImageTransformRenderer,
  imageAdjPipeline,
  renderImageTransformFrame,
} from "./transformations/transformations";
import { AIState, UIState } from "./state/state";
import { createButton } from "./ui/button";
import { createOverlayCanvas } from "./ui/canvas";

function modifyVideo(video: HTMLVideoElement) {
  if (video.dataset.filterAttached) return;
  video.dataset.filterAttached = "true";
  video.crossOrigin = "anonymous";

  const canvas = createOverlayCanvas(video);

  let renderer: any = null;
  let aiTriggered = false;

  function loop() {
    if (!document.contains(video)) {
      if (renderer) destroyImageTransformRenderer(renderer);
      return;
    }

    if (UIState.filterEnabled) {
      canvas.style.display = "block";
      video.style.opacity = "0";

      if (!renderer) {
        renderer = createImageTransformRenderer(
          canvas,
          clipToUV,
          imageAdjPipeline,
          { metrics: { enabled: false } },
        );
      }

      if (renderer && !video.paused && !video.ended) {
        if (!aiTriggered && video.readyState >= 2) {
          aiTriggered = true;
          runAIPrediction(video).catch(console.error);
        }
        renderImageTransformFrame(renderer, video, AIState.params);
      }
    } else {
      canvas.style.display = "none";
      video.style.opacity = "1";
    }
    requestAnimationFrame(loop);
  }
  loop();
}

export function initInstagramFilter() {
  createButton();
  observeVideos(modifyVideo);
}
