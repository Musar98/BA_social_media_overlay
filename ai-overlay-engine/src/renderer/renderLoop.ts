import { UIState, AIState } from "../state/state";
import { renderer } from "./renderer";
import { overlayLogger } from "../diagnostics/logger";
import { createIdentityAIParams } from "../ai/IdentityParams";

const AI_WORKER_PATH =
  "/static_resources/webworker_v1/init_script/ai-worker.iife.js";

type VideoWithFrameCallback = HTMLVideoElement & {
  requestVideoFrameCallback?: (callback: () => void) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

class RenderLoop {
  private animationId: number | null = null;
  private videoFrameCallbackId: number | null = null;
  private static workerInstance: Worker | null = null;
  private activeVideo: HTMLVideoElement | null = null;
  private stopped = true;
  private generation = 0;
  private aiTaskInFlight = false;
  private workerReady = false;
  private loopStartCount = 0;
  private loopStopCount = 0;
  private workerCreateCount = 0;
  private workerTerminateCount = 0;
  private aiTaskStartCount = 0;
  private aiTaskCompleteCount = 0;
  private workerWarmupTimer: number | null = null;

  // MobileNetV4 backbone input size
  private readonly aiInputSize = 224;

  private getWorker(): Worker | null {
    if (RenderLoop.workerInstance) {
      return RenderLoop.workerInstance;
    }

    try {
      const worker = new Worker(AI_WORKER_PATH);
      const workerGeneration = this.generation;

      RenderLoop.workerInstance = worker;
      this.workerCreateCount += 1;
      overlayLogger.info("ai-worker-created", {
        workerCreateCount: this.workerCreateCount,
        generation: this.generation,
      });

      worker.onmessage = (event) => {
        if (worker !== RenderLoop.workerInstance) {
          return;
        }

        const { aiParams, error, type } = event.data;

        if (error) {
          overlayLogger.error("ai-worker-error", { error });
        }

        if (type === "initialized") {
          this.workerReady = true;
          overlayLogger.info("ai-worker-ready");
        }

        if (aiParams && !this.stopped && workerGeneration === this.generation) {
          AIState.params = aiParams;
        }

        if (aiParams || error) {
          this.aiTaskInFlight = false;
          this.aiTaskCompleteCount += 1;
          overlayLogger.verbose("ai-task-completed", {
            aiTaskCompleteCount: this.aiTaskCompleteCount,
            generation: this.generation,
            hasError: Boolean(error),
          });
        }
      };

      // Signal initialization/warmup
      worker.postMessage({ type: "init" });
      return worker;
    } catch (err) {
      overlayLogger.error("ai-worker-create-failed", { error: String(err) });
      return null;
    }
  }

  start(
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    aiFrameInterval = 60,
  ): void {
    this.stop();

    this.activeVideo = video;
    this.stopped = false;
    this.generation += 1;
    this.aiTaskInFlight = false;
    this.workerReady = false;
    AIState.params = createIdentityAIParams();

    const loopGeneration = this.generation;
    this.loopStartCount += 1;
    overlayLogger.info("render-loop-start", {
      loopStartCount: this.loopStartCount,
      generation: this.generation,
      video: this.describeVideo(video),
    });

    renderer.init(canvas);
    renderer.resetSmoothing();
    let validFrameCount = 0;

    let canvasVisible: boolean | null = null;
    let hasRevealedOverlay = false;
    let containsCheckFrame = 0;

    const scheduleWorkerWarmup = () => {
      if (
        RenderLoop.workerInstance ||
        this.workerWarmupTimer !== null ||
        this.stopped ||
        loopGeneration !== this.generation
      ) {
        return;
      }

      this.workerWarmupTimer = window.setTimeout(() => {
        this.workerWarmupTimer = null;

        if (this.stopped || loopGeneration !== this.generation) {
          return;
        }

        this.getWorker();
      }, 0);
    };

    const triggerAI = async () => {
      if (this.aiTaskInFlight || this.stopped || loopGeneration !== this.generation) {
        return;
      }

      const currentWorker = this.getWorker();
      if (!currentWorker || !this.workerReady || video.readyState < 2) {
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

      this.aiTaskInFlight = true;
      this.aiTaskStartCount += 1;
      overlayLogger.verbose("ai-task-started", {
        aiTaskStartCount: this.aiTaskStartCount,
        generation: this.generation,
      });

      try {
        const bitmap = await createImageBitmap(video, sourceX, sourceY, sourceSize, sourceSize, {
          resizeWidth: targetSize,
          resizeHeight: targetSize,
          resizeQuality: "medium",
        });

        if (this.stopped || loopGeneration !== this.generation || this.activeVideo !== video) {
          bitmap.close();
          this.aiTaskInFlight = false;
          this.aiTaskCompleteCount += 1;
          return;
        }

        currentWorker.postMessage({ bitmap }, [bitmap]);
      } catch (err) {
        overlayLogger.error("image-bitmap-create-failed", {
          error: String(err),
        });
        if (loopGeneration === this.generation) {
          this.aiTaskInFlight = false;
          this.aiTaskCompleteCount += 1;
        }
      }
    };

    let lastFilterEnabled = UIState.filterEnabled;
    const updateVisibility = (enabled: boolean) => {
      const shouldShowCanvas = enabled && hasRevealedOverlay;

      if (canvasVisible === shouldShowCanvas) {
        return;
      }

      canvasVisible = shouldShowCanvas;

      if (shouldShowCanvas) {
        canvas.style.display = "block";
        video.style.opacity = "0";
      } else {
        canvas.style.display = "none";
        video.style.opacity = "1";
      }
    };

    updateVisibility(lastFilterEnabled);

    const loop = () => {
      if (this.stopped || loopGeneration !== this.generation) {
        return;
      }

      containsCheckFrame += 1;
      if (containsCheckFrame >= 30 && !document.contains(video)) {
        this.stop();
        return;
      }

      if (containsCheckFrame >= 30) {
        containsCheckFrame = 0;
      }

      if (UIState.filterEnabled !== lastFilterEnabled) {
        lastFilterEnabled = UIState.filterEnabled;
        updateVisibility(lastFilterEnabled);
      }

      if (UIState.filterEnabled) {
        if (!video.paused && !video.ended) {
          const renderResult = renderer.renderFrame(video, AIState.params);

          if (renderResult?.drawn) {
            if (!hasRevealedOverlay) {
              hasRevealedOverlay = true;
              updateVisibility(lastFilterEnabled);
            }

            validFrameCount += 1;
            scheduleWorkerWarmup();

            if (validFrameCount % aiFrameInterval === 0) {
              triggerAI();
            }
          }
        }
      }

      scheduleNextFrame();
    };

    const scheduleNextFrame = () => {
      if (this.stopped || loopGeneration !== this.generation) {
        return;
      }

      const frameVideo = video as VideoWithFrameCallback;

      if (typeof frameVideo.requestVideoFrameCallback === "function") {
        this.videoFrameCallbackId = frameVideo.requestVideoFrameCallback(loop);
        return;
      }

      this.animationId = requestAnimationFrame(loop);
    };

    scheduleNextFrame();
  }

  stop(): void {
    this.stopped = true;
    this.generation += 1;
    this.aiTaskInFlight = false;
    this.workerReady = false;
    this.loopStopCount += 1;

    if (this.workerWarmupTimer !== null) {
      clearTimeout(this.workerWarmupTimer);
      this.workerWarmupTimer = null;
    }

    overlayLogger.info("render-loop-stop", {
      loopStopCount: this.loopStopCount,
      generation: this.generation,
    });

    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }

    if (this.videoFrameCallbackId !== null && this.activeVideo) {
      const frameVideo = this.activeVideo as VideoWithFrameCallback;
      frameVideo.cancelVideoFrameCallback?.(this.videoFrameCallbackId);
    }

    this.animationId = null;
    this.videoFrameCallbackId = null;
    this.activeVideo = null;

    renderer.clearSourceTexture();

    if (RenderLoop.workerInstance) {
      this.workerTerminateCount += 1;
      overlayLogger.info("ai-worker-terminated", {
        workerTerminateCount: this.workerTerminateCount,
        generation: this.generation,
      });

      RenderLoop.workerInstance.terminate();
      RenderLoop.workerInstance = null;
    }

    renderer.destroy();
  }

  private describeVideo(video: HTMLVideoElement): Record<string, unknown> {
    return {
      readyState: video.readyState,
      paused: video.paused,
      ended: video.ended,
      width: video.videoWidth,
      height: video.videoHeight,
      hasSrc: Boolean(video.currentSrc || video.src || video.srcObject),
    };
  }
}

export const renderLoop = new RenderLoop();
