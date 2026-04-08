import { AIState, UIState } from "../state/state";
import { renderImageTransformFrame } from "../transformations/transformations";
import { initRenderer, cleanupRenderer } from "./renderer";

const AI_WORKER_PATH = "/static_resources/webworker_v1/init_script/aiWorker.js";

export function startRenderLoop(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  aiFrameInterval = 30,
) {
  let frameCount = 0;
  let worker: Worker | null = null;

  try {
    worker = new Worker(AI_WORKER_PATH);
    worker.onmessage = (event) => {
      const { aiParams, error } = event.data;
      if (error) {
        console.error("AI Worker error:", error);
      }
      if (aiParams) {
        const firstKey = Object.keys(aiParams)[0];
        console.log("First param:", firstKey, aiParams[firstKey])
        AIState.params = aiParams;
      }
    };
  } catch (err) {
    console.error("Failed to create AI worker:", err);
  }

  function triggerAI() {
    if (!worker || video.readyState < 2) return;

    const offscreen = new OffscreenCanvas(video.videoWidth, video.videoHeight);
    const ctx = offscreen.getContext("2d")!;
    ctx.drawImage(video, 0, 0);
    const bitmap = offscreen.transferToImageBitmap();
    worker.postMessage({ bitmap }, [bitmap]);
  }

  const renderer = initRenderer(canvas);

  function loop() {
    if (!document.contains(video)) {
      stopRenderLoop();
      worker?.terminate();
      return;
    }

    if (UIState.filterEnabled) {
      canvas.style.display = "block";
      video.style.opacity = "0";

      if (renderer && !video.paused && !video.ended) {
        frameCount++;

        if (frameCount % aiFrameInterval === 0) {
          triggerAI();
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
