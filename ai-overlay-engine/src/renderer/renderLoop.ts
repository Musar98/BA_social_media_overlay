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
  private static workerInitialized = false;
  private static workerInitializing = false;
  private activeVideo: HTMLVideoElement | null = null;
  private stopped = true;
  private generation = 0;
  private aiTaskInFlight = false;
  private loopStartCount = 0;
  private loopStopCount = 0;
  private workerCreateCount = 0;
  private workerTerminateCount = 0;
  private aiTaskStartCount = 0;
  private aiTaskCompleteCount = 0;
  private workerWarmupTimer: number | null = null;
  private pendingImmediateInferenceGeneration: number | null = null;

  // MobileNetV4 backbone input size
  private readonly aiInputSize = 224;

  private getWorker(): Worker | null {
    if (RenderLoop.workerInstance) {
      return RenderLoop.workerInstance;
    }

    try {
      const worker = new Worker(AI_WORKER_PATH);

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

        const { aiParams, error, generation, type } = event.data;

        if (error) {
          overlayLogger.error("ai-worker-error", { error });
        }

        if (type === "initialized") {
          RenderLoop.workerInitialized = true;
          RenderLoop.workerInitializing = false;
          overlayLogger.info("ai-worker-ready");

          if (
            this.pendingImmediateInferenceGeneration !== null &&
            this.pendingImmediateInferenceGeneration === this.generation
          ) {
            const pendingGeneration = this.pendingImmediateInferenceGeneration;
            this.pendingImmediateInferenceGeneration = null;
            void this.triggerAI(pendingGeneration);
          }

          return;
        }

        if (type === "prediction" && aiParams && !this.stopped && generation === this.generation) {
          AIState.params = aiParams;
        }

        if (type === "prediction" || error) {
          this.aiTaskInFlight = false;
          this.aiTaskCompleteCount += 1;
          overlayLogger.verbose("ai-task-completed", {
            aiTaskCompleteCount: this.aiTaskCompleteCount,
            generation,
            hasError: Boolean(error),
          });
        }
      };
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
    this.pendingImmediateInferenceGeneration = null;
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
        this.ensureWorkerInitialized();
      }, 0);
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

            if (validFrameCount === 1) {
              void this.triggerAI(loopGeneration, true);
            } else if ((validFrameCount - 1) % aiFrameInterval === 0) {
              void this.triggerAI(loopGeneration);
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
    this.pendingImmediateInferenceGeneration = null;
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
    renderer.destroy();
  }

  disposeWorker(): void {
    this.pendingImmediateInferenceGeneration = null;
    this.aiTaskInFlight = false;

    if (!RenderLoop.workerInstance) {
      return;
    }

    this.workerTerminateCount += 1;
    overlayLogger.info("ai-worker-terminated", {
      workerTerminateCount: this.workerTerminateCount,
      generation: this.generation,
    });

    RenderLoop.workerInstance.terminate();
    RenderLoop.workerInstance = null;
    RenderLoop.workerInitialized = false;
    RenderLoop.workerInitializing = false;
  }

  private ensureWorkerInitialized(): void {
    const worker = this.getWorker();

    if (!worker || RenderLoop.workerInitialized || RenderLoop.workerInitializing) {
      return;
    }

    RenderLoop.workerInitializing = true;
    worker.postMessage({ type: "init" });
  }

  private async triggerAI(
    loopGeneration: number,
    isImmediate = false,
  ): Promise<void> {
    if (this.aiTaskInFlight || this.stopped || loopGeneration !== this.generation) {
      return;
    }

    const currentWorker = this.getWorker();
    if (!currentWorker) {
      return;
    }

    if (!RenderLoop.workerInitialized) {
      this.ensureWorkerInitialized();

      if (isImmediate) {
        this.pendingImmediateInferenceGeneration = loopGeneration;
      }

      return;
    }

    const video = this.activeVideo;
    if (!video || video.readyState < 2 || video.paused || video.ended) {
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
      generation: loopGeneration,
      isImmediate,
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

      currentWorker.postMessage({ type: "predict", bitmap, generation: loopGeneration }, [bitmap]);
    } catch (err) {
      overlayLogger.error("image-bitmap-create-failed", {
        error: String(err),
      });
      if (loopGeneration === this.generation) {
        this.aiTaskInFlight = false;
        this.aiTaskCompleteCount += 1;
      }
    }
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
