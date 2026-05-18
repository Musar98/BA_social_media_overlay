import { overlayCanvasFactory } from "../ui/canvas";
import { renderLoop } from "../renderer/renderLoop";
import { videoResourceManager } from "./videoResourceManager";
import { overlayLogger } from "../diagnostics/logger";

class VideoModifier {
  private currentVideo: HTMLVideoElement | null = null;
  private currentCanvas: HTMLCanvasElement | null = null;
  private switchGeneration = 0;
  private readonly cleanupGapMs = 150;

  async modify(video: HTMLVideoElement): Promise<void> {
    await this.switchTo(video);
  }

  async switchTo(video: HTMLVideoElement): Promise<void> {
    if (video === this.currentVideo) return;

    this.cleanup(false);

    this.switchGeneration += 1;
    const generation = this.switchGeneration;
    videoResourceManager.enforceSingleActiveVideo(video);
    video.dataset.filterAttached = "true";
    video.crossOrigin = "anonymous";

    this.currentVideo = video;
    this.currentCanvas = overlayCanvasFactory.create(video);
    const canvas = this.currentCanvas;

    overlayLogger.info("video-switch-cleanup-gap-start", {
      cleanupGapMs: this.cleanupGapMs,
      generation,
    });

    await new Promise((resolve) => setTimeout(resolve, this.cleanupGapMs));

    if (generation !== this.switchGeneration || !document.contains(video)) {
      overlayLogger.info("video-switch-abandoned", {
        generation,
        isStale: generation !== this.switchGeneration,
        inDocument: document.contains(video),
      });

      if (this.currentVideo === video) {
        this.currentVideo.style.opacity = "1";
        this.currentVideo = null;
      }

      if (this.currentCanvas === canvas) {
        canvas.width = 1;
        canvas.height = 1;
        canvas.remove();
        this.currentCanvas = null;
      }

      return;
    }

    renderLoop.start(video, this.currentCanvas);
  }

  cleanup(invalidatePendingSwitch = true): void {
    if (invalidatePendingSwitch) {
      this.switchGeneration += 1;
    }

    if (this.currentVideo) {
      renderLoop.stop();
      this.currentVideo.style.opacity = "1";
      this.currentVideo = null;
    }

    if (this.currentCanvas) {
      this.currentCanvas.width = 1;
      this.currentCanvas.height = 1;
      this.currentCanvas.remove();
      this.currentCanvas = null;
    }
  }
}

export const videoModifier = new VideoModifier();
