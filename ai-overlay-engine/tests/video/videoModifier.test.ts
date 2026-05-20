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
    window.history.replaceState(null, "", "/");
    video = document.createElement("video");
    video.load = vi.fn();
    video.pause = vi.fn();
    video.play = vi.fn().mockResolvedValue(undefined);
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

  it("removes the previous overlay canvas when switching videos", async () => {
    const firstCanvas = document.createElement("canvas");
    const secondCanvas = document.createElement("canvas");
    const firstRemove = vi.spyOn(firstCanvas, "remove");
    const secondVideo = document.createElement("video");
    secondVideo.load = vi.fn();
    secondVideo.pause = vi.fn();
    secondVideo.play = vi.fn().mockResolvedValue(undefined);
    document.body.appendChild(secondVideo);

    vi.mocked(overlayCanvasFactory.create)
      .mockReturnValueOnce(firstCanvas)
      .mockReturnValueOnce(secondCanvas);

    await videoModifier.modify(video);
    await videoModifier.modify(secondVideo);

    expect(renderLoop.stop).toHaveBeenCalled();
    expect(firstCanvas.width).toBe(1);
    expect(firstCanvas.height).toBe(1);
    expect(firstRemove).toHaveBeenCalled();
    expect(renderLoop.start).toHaveBeenLastCalledWith(secondVideo, secondCanvas);
  });

  it("normalizes and restores active reels layout", async () => {
    window.history.replaceState(null, "", "/reels/");
    document.body.innerHTML = "";

    const main = document.createElement("main");
    const shell = document.createElement("div");
    const media = document.createElement("div");
    const canvas = document.createElement("canvas");

    media.style.height = "640px";
    media.style.overflow = "visible";
    video.style.height = "640px";
    video.style.objectFit = "contain";

    document.body.appendChild(main);
    main.appendChild(shell);
    shell.appendChild(media);
    media.appendChild(video);
    vi.mocked(overlayCanvasFactory.create).mockReturnValueOnce(canvas);

    await videoModifier.modify(video);

    expect(media.style.height).toBe("100vh");
    expect(media.style.overflow).toBe("hidden");
    expect(video.style.height).toBe("100%");
    expect(video.style.objectFit).toBe("cover");
    expect(canvas.style.position).toBe("absolute");
    expect(canvas.style.height).toBe("100%");

    videoModifier.cleanup();

    expect(media.style.height).toBe("640px");
    expect(media.style.overflow).toBe("visible");
    expect(video.style.height).toBe("640px");
    expect(video.style.objectFit).toBe("contain");
  });

  it("soft-retires the previous active video when a new video becomes active", async () => {
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
    expect(video.getAttribute("src")).toBe("https://example.com/previous.mp4");
    expect(video.preload).not.toBe("none");
    expect(video.load).not.toHaveBeenCalled();
    expect(nextVideo.pause).not.toHaveBeenCalled();
  });

  it("keeps retained video source state intact", async () => {
    const source = document.createElement("source");
    source.setAttribute("src", "https://example.com/source.mp4");
    source.setAttribute("type", "video/mp4");
    video.appendChild(source);
    video.src = "https://example.com/previous.mp4";
    video.srcObject = {} as MediaProvider;
    video.preload = "auto";

    const nextVideo = document.createElement("video");
    nextVideo.load = vi.fn();
    nextVideo.pause = vi.fn();
    nextVideo.play = vi.fn().mockResolvedValue(undefined);
    document.body.appendChild(nextVideo);

    await videoModifier.modify(video);
    await videoModifier.modify(nextVideo);

    expect(video.getAttribute("src")).toBe("https://example.com/previous.mp4");
    expect(video.srcObject).toBeTruthy();
    expect(source.getAttribute("src")).toBe("https://example.com/source.mp4");
    expect(source.getAttribute("type")).toBe("video/mp4");
    expect(video.preload).toBe("auto");
  });

  it("reactivates a retained previous video without reloading it", async () => {
    const nextVideo = document.createElement("video");
    nextVideo.load = vi.fn();
    nextVideo.pause = vi.fn();
    nextVideo.play = vi.fn().mockResolvedValue(undefined);
    document.body.appendChild(nextVideo);

    video.src = "https://example.com/previous.mp4";

    await videoModifier.modify(video);
    vi.mocked(video.play).mockClear();

    await videoModifier.modify(nextVideo);
    await videoModifier.modify(video);

    expect(video.play).toHaveBeenCalled();
    expect(video.load).not.toHaveBeenCalled();
    expect(video.getAttribute("src")).toBe("https://example.com/previous.mp4");
  });

  it("hard-releases older videos outside the retention window", async () => {
    const videos = [
      video,
      document.createElement("video"),
      document.createElement("video"),
      document.createElement("video"),
    ];

    videos.forEach((item, index) => {
      item.src = `https://example.com/video-${index}.mp4`;
      item.load = vi.fn();
      item.pause = vi.fn();
      item.play = vi.fn().mockResolvedValue(undefined);

      if (!item.isConnected) {
        document.body.appendChild(item);
      }
    });

    for (const item of videos) {
      await videoModifier.modify(item);
    }

    expect(videos[0].pause).toHaveBeenCalled();
    expect(videos[0].getAttribute("src")).toBeNull();
    expect(videos[0].preload).toBe("none");
    expect(videos[0].load).toHaveBeenCalled();

    expect(videos[1].getAttribute("src")).toBe("https://example.com/video-1.mp4");
    expect(videos[2].getAttribute("src")).toBe("https://example.com/video-2.mp4");
    expect(videos[3].getAttribute("src")).toBe("https://example.com/video-3.mp4");
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
