// @vitest-environment jsdom
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { videoModifier } from "../../src/video/videoModifier";
import { overlayCanvasFactory } from "../../src/ui/canvas";
import { renderLoop } from "../../src/renderer/renderLoop";
import { videoResourceManager } from "../../src/video/videoResourceManager";

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
    document.body.innerHTML = "";
    video = document.createElement("video");
    document.body.appendChild(video);
  });

  afterEach(() => {
    videoModifier.cleanup();
    videoResourceManager.forgetActiveVideo(null);
    document.body.innerHTML = "";
  });

  it("modifies video element correctly", async () => {
    await videoModifier.modify(video);

    expect(video.dataset.filterAttached).toBe("true");
    expect(video.crossOrigin).toBe("anonymous");
    expect(overlayCanvasFactory.create).toHaveBeenCalledWith(video);
    expect(renderLoop.start).toHaveBeenCalled();
  });

  it("attaches the canvas before the cleanup gap finishes", async () => {
    const modifyPromise = videoModifier.modify(video);

    expect(overlayCanvasFactory.create).toHaveBeenCalledWith(video);
    expect(renderLoop.start).not.toHaveBeenCalled();

    await modifyPromise;

    expect(renderLoop.start).toHaveBeenCalled();
  });

  it("cleans up correctly", async () => {
    await videoModifier.modify(video);
    videoModifier.cleanup();

    expect(renderLoop.stop).toHaveBeenCalled();
    // In our implementation, cleanup doesn't remove the dataset, but sets currentVideo to null.
    // We can verify that another call to modify works.
    await videoModifier.modify(video);
    expect(overlayCanvasFactory.create).toHaveBeenCalledTimes(2);
  });

  it("releases the previous active video when a new video becomes active", async () => {
    const nextVideo = document.createElement("video");
    nextVideo.load = vi.fn();
    nextVideo.pause = vi.fn();
    nextVideo.play = vi.fn().mockResolvedValue(undefined);
    document.body.appendChild(nextVideo);

    video.src = "https://example.com/previous.mp4";
    video.load = vi.fn();
    video.pause = vi.fn();

    await videoModifier.modify(video);
    await videoModifier.modify(nextVideo);

    expect(video.pause).toHaveBeenCalled();
    expect(video.getAttribute("src")).toBeNull();
    expect(video.preload).toBe("none");
    expect(video.load).toHaveBeenCalled();
    expect(nextVideo.pause).not.toHaveBeenCalled();
  });

  it("does not pause videos before an active video is known", () => {
    video.src = "https://example.com/current.mp4";
    video.pause = vi.fn();
    video.load = vi.fn();

    videoResourceManager.enforceSingleActiveVideo(null);

    expect(video.pause).not.toHaveBeenCalled();
    expect(video.getAttribute("src")).toBe("https://example.com/current.mp4");
    expect(video.load).not.toHaveBeenCalled();
  });
});
