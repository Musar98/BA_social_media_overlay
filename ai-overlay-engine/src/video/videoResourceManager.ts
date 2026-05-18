import { overlayLogger } from "../diagnostics/logger";

type SourceChildState = {
  element: HTMLSourceElement;
  src: string | null;
  type: string | null;
};

type ReleasedVideoState = {
  src: string;
  srcAttribute: string | null;
  preloadAttribute: string | null;
  currentTime: number;
  wasPaused: boolean;
  srcObject: MediaProvider | null;
  sources: SourceChildState[];
};

class VideoResourceManager {
  private readonly releasedVideos = new WeakMap<
    HTMLVideoElement,
    ReleasedVideoState
  >();
  private activeVideo: HTMLVideoElement | null = null;

  enforceSingleActiveVideo(activeVideo: HTMLVideoElement | null): void {
    if (!activeVideo) {
      return;
    }

    const previousActiveVideo = this.activeVideo;
    this.activeVideo = activeVideo;

    this.logVideoSnapshot("single-active-video-enforced", activeVideo);

    this.restoreVideo(activeVideo);

    if (previousActiveVideo && previousActiveVideo !== activeVideo) {
      this.releaseVideo(previousActiveVideo);
    }
  }

  forgetActiveVideo(video: HTMLVideoElement | null): void {
    if (!video || this.activeVideo === video) {
      this.activeVideo = null;
    }
  }

  restoreVideo(video: HTMLVideoElement): void {
    const state = this.releasedVideos.get(video);

    if (!state) {
      video.preload = video.preload || "auto";
      this.logVideoSnapshot("active-video-already-attached", video);
      return;
    }

    if (state.preloadAttribute === null) {
      video.removeAttribute("preload");
    } else {
      video.setAttribute("preload", state.preloadAttribute);
    }

    if (state.srcAttribute === null) {
      video.removeAttribute("src");
    } else {
      video.setAttribute("src", state.srcAttribute);
    }

    if (state.src && !video.currentSrc && !video.src) {
      video.src = state.src;
    }

    video.srcObject = state.srcObject;

    state.sources.forEach((sourceState) => {
      if (sourceState.src === null) {
        sourceState.element.removeAttribute("src");
      } else {
        sourceState.element.setAttribute("src", sourceState.src);
      }

      if (sourceState.type === null) {
        sourceState.element.removeAttribute("type");
      } else {
        sourceState.element.setAttribute("type", sourceState.type);
      }
    });

    this.load(video);

    if (state.currentTime > 0) {
      try {
        video.currentTime = state.currentTime;
      } catch {
        // Some streams reject currentTime restores until metadata is available.
      }
    }

    this.play(video);

    this.releasedVideos.delete(video);
    this.logVideoSnapshot("active-video-restored", video);
  }

  releaseVideo(video: HTMLVideoElement): void {
    if (!this.releasedVideos.has(video)) {
      this.releasedVideos.set(video, {
        src: video.currentSrc || video.src,
        srcAttribute: video.getAttribute("src"),
        preloadAttribute: video.getAttribute("preload"),
        currentTime: video.currentTime,
        wasPaused: video.paused,
        srcObject: video.srcObject,
        sources: Array.from(video.querySelectorAll("source")).map((source) => ({
          element: source,
          src: source.getAttribute("src"),
          type: source.getAttribute("type"),
        })),
      });
    }

    this.pause(video);
    video.preload = "none";
    video.removeAttribute("src");
    video.srcObject = null;

    video.querySelectorAll("source").forEach((source) => {
      source.removeAttribute("src");
    });

    this.load(video);
    this.logVideoSnapshot("previous-video-released", video);
  }

  private logVideoSnapshot(
    event: string,
    activeVideo: HTMLVideoElement,
  ): void {
    const videos = Array.from(document.querySelectorAll("video"));

    overlayLogger.verbose("video-resources", {
      event,
      videoCount: videos.length,
      activeIndex: videos.indexOf(activeVideo),
      active: this.describeVideo(activeVideo),
      videos: videos.map((video, index) => ({
        index,
        isActive: video === activeVideo,
        ...this.describeVideo(video),
      })),
    });
  }

  private describeVideo(video: HTMLVideoElement): Record<string, unknown> {
    return {
      readyState: video.readyState,
      paused: video.paused,
      ended: video.ended,
      width: video.videoWidth,
      height: video.videoHeight,
      hasSrc: Boolean(video.currentSrc || video.src || video.srcObject),
      preload: video.preload,
    };
  }

  private pause(video: HTMLVideoElement): void {
    try {
      video.pause();
    } catch {
      // Some test/browser shims do not implement media controls fully.
    }
  }

  private load(video: HTMLVideoElement): void {
    try {
      video.load();
    } catch {
      // load() is best-effort; removing src/srcObject is the important part.
    }
  }

  private play(video: HTMLVideoElement): void {
    try {
      video.play().catch(() => {
        // The host page may resume playback itself if autoplay is blocked.
      });
    } catch {
      // Ignore synchronous play() failures from browser policy or shims.
    }
  }
}

export const videoResourceManager = new VideoResourceManager();
