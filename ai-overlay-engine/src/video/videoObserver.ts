export function observeVideos(onVideoAdded: (v: HTMLVideoElement) => void) {
  const observer = new MutationObserver((muts) => {
    muts.forEach((m) => {
      m.addedNodes.forEach((node) => {
        if (node instanceof HTMLVideoElement) onVideoAdded(node);
        else if (node instanceof Element)
          node.querySelectorAll("video").forEach(onVideoAdded);
      });
    });
  });

  observer.observe(document.body, { childList: true, subtree: true });
  document.querySelectorAll("video").forEach(onVideoAdded);
}
