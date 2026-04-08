import { UIState, AIState } from "../state/state";
import { renderImageTransformFrame } from "../transformations/transformations";
import {
  createImageTransformRenderer,
  destroyImageTransformRenderer,
} from "../transformations/transformations";
import { CLIP_TO_UV } from "../transformations/ClipToUv";
import { IMAGE_ADJUSTMENTS_PIPELINE } from "../transformations/ImageAdjustmentsPipeline";
import { runAIPrediction } from "../ai/ai-engine";

let renderer: any = null;
let animationId: number | null = null;

export function startRenderLoop(
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement
) {
  let aiTriggered = false;

  // 🔥 ensure only ONE renderer exists
  if (renderer) {
    cleanupRenderer();
  }

  renderer = createImageTransformRenderer(
      canvas,
      CLIP_TO_UV,
      IMAGE_ADJUSTMENTS_PIPELINE,
      { metrics: { enabled: false } }
  );

  function loop() {
    // stop if video removed
    if (!document.contains(video)) {
      cleanupRenderer();
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

  cleanupRenderer();
}

function cleanupRenderer() {
  if (!renderer) return;

  // 🔥 CRITICAL: force WebGL context release
  const gl = renderer?.gl;
  if (gl) {
    const ext = gl.getExtension("WEBGL_lose_context");
    ext?.loseContext();
  }

  destroyImageTransformRenderer(renderer);
  renderer = null;
}