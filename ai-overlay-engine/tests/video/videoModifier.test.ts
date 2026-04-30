// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from "vitest";
import { videoModifier } from "../../src/video/videoModifier";
import { overlayCanvasFactory } from "../../src/ui/canvas";
import { renderLoop } from "../../src/renderer/renderLoop";

vi.mock("../../src/ui/canvas", () => ({
  overlayCanvasFactory: {
    create: vi.fn().mockReturnValue(document.createElement("canvas")),
  },
}));

vi.mock("../../src/renderer/renderLoop", () => ({
  renderLoop: {
    start: vi.fn(),
    stop: vi.fn(),
  },
}));

describe("VideoModifier", () => {
  let video: HTMLVideoElement;

  beforeEach(() => {
    vi.clearAllMocks();
    video = document.createElement("video");
    document.body.appendChild(video);
  });

  it("modifies video element correctly", () => {
    videoModifier.modify(video);

    expect(video.dataset.filterAttached).toBe("true");
    expect(video.crossOrigin).toBe("anonymous");
    expect(overlayCanvasFactory.create).toHaveBeenCalledWith(video);
    expect(renderLoop.start).toHaveBeenCalled();
  });

  it("cleans up correctly", () => {
    videoModifier.modify(video);
    videoModifier.cleanup();

    expect(renderLoop.stop).toHaveBeenCalled();
    // In our implementation, cleanup doesn't remove the dataset, but sets currentVideo to null.
    // We can verify that another call to modify works.
    videoModifier.modify(video);
    expect(overlayCanvasFactory.create).toHaveBeenCalledTimes(2);
  });
});
