import { videoModifier } from "./videoModifier";

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
          if (node instanceof HTMLVideoElement) {
            this.intersectionObserver.observe(node);
          } else if (node instanceof Element) {
            node.querySelectorAll("video").forEach((v) => {
              this.intersectionObserver.observe(v);
            });
          }
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
}

export const videoObserver = new VideoObserver();
