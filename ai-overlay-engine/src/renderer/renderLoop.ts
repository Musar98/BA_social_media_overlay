import { UIState, AIState } from "../state/state";
import { renderer } from "./renderer";

const AI_WORKER_PATH =
  "/static_resources/webworker_v1/init_script/ai-worker.iife.js";

class RenderLoop {
  private animationId: number | null = null;
  private worker: Worker | null = null;

  // MobileNetV4 backbone input size
  private readonly aiInputSize = 224;

  start(
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    aiFrameInterval = 120,
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

      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;

      if (!videoWidth || !videoHeight) {
        return;
      }

      const targetSize = this.aiInputSize;

      const sourceSize = Math.min(videoWidth, videoHeight);
      const sourceX = (videoWidth - sourceSize) / 2;
      const sourceY = (videoHeight - sourceSize) / 2;

      const offscreen = new OffscreenCanvas(targetSize, targetSize);
      const ctx = offscreen.getContext("2d", {
        alpha: false,
        desynchronized: true,
      });

      if (!ctx) {
        return;
      }

      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";

      ctx.drawImage(
        video,
        sourceX,
        sourceY,
        sourceSize,
        sourceSize,
        0,
        0,
        targetSize,
        targetSize,
      );

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