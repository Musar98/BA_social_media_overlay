import { UIState, AIState } from "../state/state";
import { renderer } from "./renderer";

const AI_WORKER_PATH =
  "/static_resources/webworker_v1/init_script/ai-worker.iife.js";

class RenderLoop {
  private animationId: number | null = null;
  private static workerInstance: Worker | null = null;

  // MobileNetV4 backbone input size
  private readonly aiInputSize = 224;

  private getWorker(): Worker | null {
    if (RenderLoop.workerInstance) {
      return RenderLoop.workerInstance;
    }

    try {
      RenderLoop.workerInstance = new Worker(AI_WORKER_PATH);
      RenderLoop.workerInstance.onmessage = (event) => {
        const { aiParams, error, type } = event.data;

        if (error) {
          console.error("AI Worker error:", error);
        }

        if (aiParams) {
          AIState.params = aiParams;
        }

        if (type === "initialized") {
          console.info("AI Worker pre-warmed and ready");
        }
      };

      // Signal initialization/warmup
      RenderLoop.workerInstance.postMessage({ type: "init" });
      return RenderLoop.workerInstance;
    } catch (err) {
      console.error("Failed to create AI worker:", err);
      return null;
    }
  }

  start(
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    aiFrameInterval = 120,
  ): void {
    renderer.init(canvas);
    let frameCount = 0;

    const worker = this.getWorker();

    const triggerAI = async () => {
      const currentWorker = this.getWorker();
      if (!currentWorker || video.readyState < 2) {
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

      try {
        const bitmap = await createImageBitmap(video, sourceX, sourceY, sourceSize, sourceSize, {
          resizeWidth: targetSize,
          resizeHeight: targetSize,
          resizeQuality: "medium",
        });

        currentWorker.postMessage({ bitmap }, [bitmap]);
      } catch (err) {
        console.error("Failed to create ImageBitmap:", err);
      }
    };

    let lastFilterEnabled = UIState.filterEnabled;
    const updateVisibility = (enabled: boolean) => {
      if (enabled) {
        canvas.style.display = "block";
        video.style.opacity = "0";
      } else {
        canvas.style.display = "none";
        video.style.opacity = "1";
      }
    };

    updateVisibility(lastFilterEnabled);

    const loop = () => {
      if (!document.contains(video)) {
        this.stop();
        return;
      }

      if (UIState.filterEnabled !== lastFilterEnabled) {
        lastFilterEnabled = UIState.filterEnabled;
        updateVisibility(lastFilterEnabled);
      }

      if (UIState.filterEnabled) {
        if (!video.paused && !video.ended) {
          frameCount++;

          if (frameCount % aiFrameInterval === 0) {
            triggerAI();
          }

          renderer.renderFrame(video, AIState.params);
        }
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

    // We no longer terminate the worker here to persist ONNX state
    // RenderLoop.workerInstance?.terminate();

    renderer.destroy();
  }
}

export const renderLoop = new RenderLoop();