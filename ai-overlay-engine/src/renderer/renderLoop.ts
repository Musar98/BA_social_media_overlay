import { UIState, AIState } from "../state/state";
import { renderer } from "./renderer";
import { AIParams } from "../ai/Types";

const AI_WORKER_PATH =
  "/static_resources/webworker_v1/init_script/ai-worker.iife.js";

class RenderLoop {
  private animationId: number | null = null;
  private worker: Worker | null = null;

  start(
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    aiFrameInterval = 120, //TODO swap values higher to 60/90/120
  ): void {
    renderer.init(canvas);
    let frameCount = 0;

    try {
      this.worker = new Worker(AI_WORKER_PATH);
      this.worker.onmessage = (event) => {
        const { aiParams, error } = event.data;
        if (error) {
          console.error("AI Worker error:", error);
        }
        if (aiParams) {
          AIState.params = aiParams;
        }
      };
    } catch (err) {
      console.error("Failed to create AI worker:", err);
    }

    const triggerAI = () => {
      if (!this.worker || video.readyState < 2) {
        return;
      }

      const offscreen = new OffscreenCanvas(
        video.videoWidth,
        video.videoHeight,
      );
      const ctx = offscreen.getContext("2d")!;
      ctx.drawImage(video, 0, 0);
      const bitmap = offscreen.transferToImageBitmap();
      this.worker.postMessage({ bitmap }, [bitmap]);
    };

    const loop = () => {
      if (!document.contains(video)) {
        this.stop();
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

          renderer.renderFrame(video, AIState.params);
        }
      } else {
        canvas.style.display = "none";
        video.style.opacity = "1";
      }

      this.animationId = requestAnimationFrame(loop);
    };

    loop();
  }

  stop(): void {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    this.animationId = null;

    this.worker?.terminate();
    this.worker = null;

    renderer.destroy();
  }
}

export const renderLoop = new RenderLoop();
