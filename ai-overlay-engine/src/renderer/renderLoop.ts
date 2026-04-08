import { AIState, UIState } from "../state/state";
import { renderImageTransformFrame } from "../transformations/transformations";
import { runAIPrediction } from "../ai/ai-engine";
import { destroyRenderer, initRenderer } from "./renderer";

let animationId: number | null = null;

export function startRenderLoop(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
) {
  let aiTriggered = false;

  const renderer = initRenderer(canvas);

  function loop() {
    if (!document.contains(video)) {
      stopRenderLoop();
      return;
    }

    if (UIState.filterEnabled) {
      canvas.style.display = "block";
      video.style.opacity = "0";

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

    animationId = requestAnimationFrame(loop);
  }

  loop();
}

export function stopRenderLoop() {
  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  destroyRenderer();
}
