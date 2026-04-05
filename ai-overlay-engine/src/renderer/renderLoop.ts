import { UIState, AIState } from "../state/state";
import { renderImageTransformFrame } from "../transformations/transformations";
import { initRenderer, cleanupRenderer } from "./renderer";
import { runAIPrediction } from "../ai/ai-engine";

export function startRenderLoop(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
) {
  let renderer: any = null;
  let aiTriggered = false;

  function loop() {
    if (!document.contains(video)) {
      cleanupRenderer(renderer);
      return;
    }

    if (UIState.filterEnabled) {
      canvas.style.display = "block";
      video.style.opacity = "0";

      renderer = initRenderer(renderer, canvas);

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
