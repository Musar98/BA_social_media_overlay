import { overlayCanvasFactory } from "../ui/canvas";
import { renderLoop } from "../renderer/renderLoop";

class VideoModifier {
  private currentVideo: HTMLVideoElement | null = null;
  private currentCanvas: HTMLCanvasElement | null = null;

  modify(video: HTMLVideoElement): void {
    if (video === this.currentVideo) return;

    document.querySelectorAll("video").forEach((v) => {
      if (v !== video) {
        (v as HTMLVideoElement).dataset.filterAttached = "true";
      }
    });

    video.dataset.filterAttached = "true";
    video.crossOrigin = "anonymous";

    this.currentVideo = video;
    this.currentCanvas = overlayCanvasFactory.create(video);

    renderLoop.start(video, this.currentCanvas);
  }

  cleanup(): void {
    if (this.currentVideo) {
      renderLoop.stop();
      this.currentVideo = null;
    }

    if (this.currentCanvas) {
      this.currentCanvas.remove();
      this.currentCanvas = null;
    }
  }
}

export const videoModifier = new VideoModifier();
