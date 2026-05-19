class OverlayCanvasFactory {
  create(video: HTMLVideoElement): HTMLCanvasElement {
    const canvas = document.createElement("canvas");

    canvas.style.position = "absolute";
    canvas.style.inset = "0";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.pointerEvents = "none";
    canvas.style.display = "none";
    canvas.style.zIndex = "0";

    video.insertAdjacentElement("afterend", canvas);

    return canvas;
  }
}

export const overlayCanvasFactory = new OverlayCanvasFactory();
