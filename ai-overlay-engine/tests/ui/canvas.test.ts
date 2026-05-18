// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { overlayCanvasFactory } from "../../src/ui/canvas";

describe("OverlayCanvasFactory", () => {
  it("creates a full-cover overlay canvas", () => {
    const video = document.createElement("video");
    document.body.appendChild(video);

    const canvas = overlayCanvasFactory.create(video);

    expect(canvas.style.position).toBe("absolute");
    expect(canvas.style.inset).toBe("0");
    expect(canvas.style.width).toBe("100%");
    expect(canvas.style.height).toBe("100%");
    expect(canvas.style.aspectRatio).toBe("");
    expect(canvas.nextElementSibling).toBeNull();
    expect(video.nextElementSibling).toBe(canvas);
  });
});
