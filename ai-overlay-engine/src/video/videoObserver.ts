import { modifyVideo, cleanupCurrentVideo } from "./videoModifier";

let activeVideo: HTMLVideoElement | null = null;

const intersectionObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      const video = entry.target as HTMLVideoElement;

      if (entry.isIntersecting && entry.intersectionRatio > 0.6) {
        if (activeVideo !== video) {
          cleanupCurrentVideo();

          activeVideo = video;
          modifyVideo(video);
        }
      }
    });
  },
  {
    threshold: [0.6],
  },
);

export function observeVideos() {
  const mutationObserver = new MutationObserver((muts) => {
    muts.forEach((m) => {
      m.addedNodes.forEach((node) => {
        if (node instanceof HTMLVideoElement) {
          intersectionObserver.observe(node);
        } else if (node instanceof Element) {
          node.querySelectorAll("video").forEach((v) => {
            intersectionObserver.observe(v);
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
    intersectionObserver.observe(v);
  });
}
