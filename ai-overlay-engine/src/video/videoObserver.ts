import { videoModifier } from "./videoModifier";
import { videoResourceManager } from "./videoResourceManager";

class VideoObserver {
  private activeVideo: HTMLVideoElement | null = null;
  private readonly intersectionObserver: IntersectionObserver;

  constructor() {
    this.intersectionObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const video = entry.target as HTMLVideoElement;

          if (entry.isIntersecting && entry.intersectionRatio > 0.6) {
            if (this.activeVideo !== video) {
              videoModifier.cleanup();

              this.activeVideo = video;
              videoModifier.modify(video);
            }
          }
        });
      },
      {
        threshold: [0.6],
      },
    );
  }

  observe(): void {
    const mutationObserver = new MutationObserver((muts) => {
      muts.forEach((m) => {
        m.addedNodes.forEach((node) => {
          this.observeVideosInNode(node);
        });

        m.removedNodes.forEach((node) => {
          this.unobserveVideosInNode(node);
        });
      });
    });

    mutationObserver.observe(document.body, {
      childList: true,
      subtree: true,
    });

    document.querySelectorAll("video").forEach((v) => {
      this.intersectionObserver.observe(v);
    });
  }

  private observeVideosInNode(node: Node): void {
    if (node instanceof HTMLVideoElement) {
      this.intersectionObserver.observe(node);
      return;
    }

    if (node instanceof Element) {
      node.querySelectorAll("video").forEach((video) => {
        this.intersectionObserver.observe(video);
      });
    }
  }

  private unobserveVideosInNode(node: Node): void {
    if (node instanceof HTMLVideoElement) {
      this.unobserveVideo(node);
      return;
    }

    if (node instanceof Element) {
      node.querySelectorAll("video").forEach((video) => {
        this.unobserveVideo(video);
      });
    }
  }

  private unobserveVideo(video: HTMLVideoElement): void {
    this.intersectionObserver.unobserve(video);

    if (this.activeVideo === video) {
      videoModifier.cleanup();
      videoResourceManager.forgetActiveVideo(video);
      this.activeVideo = null;
      videoResourceManager.enforceSingleActiveVideo(null);
    }
  }
}

export const videoObserver = new VideoObserver();
