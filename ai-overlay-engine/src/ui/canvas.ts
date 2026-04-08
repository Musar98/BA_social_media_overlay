export function createOverlayCanvas(video: HTMLVideoElement) {
  const canvas = document.createElement("canvas");

  canvas.style.position = "absolute";
  canvas.style.top = "0";
  canvas.style.left = "50%";
  canvas.style.height = "100%";
  canvas.style.aspectRatio = "9 / 16";
  canvas.style.transform = "translateX(-50%)";
  canvas.style.pointerEvents = "none";
  canvas.style.display = "none";
  canvas.style.zIndex = "0";

  video.insertAdjacentElement("afterend", canvas);

  return canvas;
}
