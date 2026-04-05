export function createOverlayCanvas(video: HTMLVideoElement) {
  const canvas = document.createElement("canvas");

  canvas.style.cssText = `
    position:absolute;
    top:0;
    left:0;
    width:100%;
    height:100%;
    pointer-events:none;
    display:none;
    z-index:0;
  `;

  video.insertAdjacentElement("afterend", canvas);

  return canvas;
}
