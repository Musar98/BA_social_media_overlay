import { overlayCanvasFactory } from "../ui/canvas";
import { renderLoop } from "../renderer/renderLoop";
import { videoResourceManager } from "./videoResourceManager";
import { overlayLogger } from "../diagnostics/logger";

type RestorableLayoutProperty =
  | "backgroundColor"
  | "height"
  | "inset"
  | "maxHeight"
  | "minHeight"
  | "objectFit"
  | "overflow"
  | "position"
  | "width";

type LayoutSnapshot = Partial<Record<RestorableLayoutProperty, string>>;

type LayoutElementSnapshot = {
  element: HTMLElement;
  snapshot: LayoutSnapshot;
};

type ReelsLayoutHandle = {
  restore: () => void;
};

const REELS_PATH_PATTERN = /^\/(?:reels?|reel)(?:\/|$)/;
const MAX_REELS_ANCESTORS = 6;
const FULL_VIEWPORT_HEIGHT = "100vh";
const VIDEO_LAYOUT_PROPERTIES: RestorableLayoutProperty[] = [
  "backgroundColor",
  "height",
  "objectFit",
  "width",
];
const CANVAS_LAYOUT_PROPERTIES: RestorableLayoutProperty[] = [
  "backgroundColor",
  "height",
  "inset",
  "objectFit",
  "position",
  "width",
];
const ANCESTOR_LAYOUT_PROPERTIES: RestorableLayoutProperty[] = [
  "backgroundColor",
  "height",
  "maxHeight",
  "minHeight",
  "overflow",
  "position",
  "width",
];

class ReelsLayoutNormalizer {
  apply(
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
  ): ReelsLayoutHandle | null {
    if (!this.shouldNormalize(video)) {
      return null;
    }

    const snapshots: LayoutElementSnapshot[] = [
      {
        element: video,
        snapshot: this.capture(video.style, VIDEO_LAYOUT_PROPERTIES),
      },
      {
        element: canvas,
        snapshot: this.capture(canvas.style, CANVAS_LAYOUT_PROPERTIES),
      },
    ];
    const ancestors = this.getTargetAncestors(video);

    for (const ancestor of ancestors) {
      snapshots.push({
        element: ancestor,
        snapshot: this.capture(ancestor.style, ANCESTOR_LAYOUT_PROPERTIES),
      });
    }

    for (const ancestor of ancestors) {
      ancestor.style.width = "100%";
      ancestor.style.height = FULL_VIEWPORT_HEIGHT;
      ancestor.style.minHeight = FULL_VIEWPORT_HEIGHT;
      ancestor.style.maxHeight = "none";
      ancestor.style.overflow = "hidden";
      ancestor.style.backgroundColor = "black";

      if (window.getComputedStyle(ancestor).position === "static") {
        ancestor.style.position = "relative";
      }
    }

    video.style.width = "100%";
    video.style.height = "100%";
    video.style.objectFit = "cover";
    video.style.backgroundColor = "black";

    canvas.style.position = "absolute";
    canvas.style.inset = "0";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.objectFit = "cover";
    canvas.style.backgroundColor = "black";

    return {
      restore: () => {
        for (const { element, snapshot } of snapshots.reverse()) {
          this.restore(element.style, snapshot);
        }
      },
    };
  }

  private shouldNormalize(video: HTMLVideoElement): boolean {
    if (typeof window === "undefined") {
      return false;
    }

    if (REELS_PATH_PATTERN.test(window.location.pathname)) {
      return true;
    }

    const viewportWidth =
      window.innerWidth || document.documentElement.clientWidth;
    const viewportHeight =
      window.innerHeight || document.documentElement.clientHeight;

    return (
      viewportHeight > viewportWidth &&
      video.videoHeight > video.videoWidth &&
      video.getBoundingClientRect().width >= viewportWidth * 0.75
    );
  }

  private getTargetAncestors(video: HTMLVideoElement): HTMLElement[] {
    const ancestors: HTMLElement[] = [];
    let current = video.parentElement;

    while (
      current &&
      current !== document.body &&
      ancestors.length < MAX_REELS_ANCESTORS
    ) {
      ancestors.push(current);

      if (
        current.tagName.toLowerCase() === "main" ||
        current.getAttribute("role") === "main"
      ) {
        break;
      }

      current = current.parentElement;
    }

    return ancestors;
  }

  private capture(
    style: CSSStyleDeclaration,
    properties: RestorableLayoutProperty[],
  ): LayoutSnapshot {
    return properties.reduce<LayoutSnapshot>((snapshot, property) => {
      snapshot[property] = style[property];
      return snapshot;
    }, {});
  }

  private restore(style: CSSStyleDeclaration, snapshot: LayoutSnapshot): void {
    for (const [property, value] of Object.entries(snapshot) as [
      RestorableLayoutProperty,
      string,
    ][]) {
      style[property] = value;
    }
  }
}

const reelsLayoutNormalizer = new ReelsLayoutNormalizer();

class VideoModifier {
  private currentVideo: HTMLVideoElement | null = null;
  private currentCanvas: HTMLCanvasElement | null = null;
  private currentLayout: ReelsLayoutHandle | null = null;
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
    this.currentLayout = reelsLayoutNormalizer.apply(video, canvas);
    const layout = this.currentLayout;

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

      if (layout && this.currentLayout === layout) {
        layout.restore();
        this.currentLayout = null;
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

    if (this.currentLayout) {
      this.currentLayout.restore();
      this.currentLayout = null;
    }
  }
}

export const videoModifier = new VideoModifier();
