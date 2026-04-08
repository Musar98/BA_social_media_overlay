import { UIState, AIState } from "../state/state";
import { renderImageTransformFrame } from "../transformations/transformations";
import { initRenderer, destroyRenderer } from "./renderer";

//TODO Remove after debugging
let lastLoggedParams;

const AI_WORKER_PATH = "/static_resources/webworker_v1/init_script/aiWorker.js";

let animationId: number | null = null;
let worker: Worker | null = null;

export function startRenderLoop(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  aiFrameInterval = 30,
) {
  const renderer = initRenderer(canvas);
  let frameCount = 0;

  try {
    worker = new Worker(AI_WORKER_PATH);
    worker.onmessage = (event) => {
      const { aiParams, error } = event.data;
      if (error) console.error("AI Worker error:", error);
      if (aiParams) AIState.params = aiParams;
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

  function loop() {
    if (!document.contains(video)) {
      stopRenderLoop();
      return;
    }

    if (UIState.filterEnabled) {
      canvas.style.display = "block";
      video.style.opacity = "0";

      if (!video.paused && !video.ended) {
        frameCount++;

        if (frameCount % aiFrameInterval === 0) {
          triggerAI();
        }

        const params = AIState.params;

        //TODO Remove after debugging
        if (
          !lastLoggedParams ||
          params?.sharp !== lastLoggedParams.sharp ||
          params?.exposure !== lastLoggedParams.exposure ||
          params?.contrast !== lastLoggedParams.contrast ||
          params?.saturation !== lastLoggedParams.saturation ||
          params?.blur !== lastLoggedParams.blur
        ) {
          console.log("[Render Loop] Rendering with params:", [
            params?.sharp,
            params?.exposure,
            params?.contrast,
            params?.saturation,
            params?.blur,
          ]);
          lastLoggedParams = { ...params }; // shallow copy
        }
        //END OF to remove

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
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;

  worker?.terminate();
  worker = null;

  destroyRenderer();
}
