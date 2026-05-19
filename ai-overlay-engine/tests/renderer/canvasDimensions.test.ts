// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SourceCanvasDimensionCache } from "../../src/renderer/canvasDimensions";

describe("SourceCanvasDimensionCache", () => {
  beforeEach(() => {
    // @ts-ignore
    global.ResizeObserver = undefined;
    vi.spyOn(performance, "now").mockReturnValue(1000);
    Object.defineProperty(window, "devicePixelRatio", {
      configurable: true,
      value: 2,
    });
  });

  it("caches displayed source dimensions after initialization", () => {
    const cache = new SourceCanvasDimensionCache();
    const video = document.createElement("video");
    const getBoundingClientRect = vi.fn(() => ({
      width: 300,
      height: 500,
      top: 0,
      left: 0,
      right: 300,
      bottom: 500,
      x: 0,
      y: 0,
      toJSON: () => {},
    }));

    video.getBoundingClientRect = getBoundingClientRect;

    expect(cache.get(video, 720, 1280)).toEqual({ width: 600, height: 1000 });
    expect(cache.get(video, 720, 1280)).toEqual({ width: 600, height: 1000 });
    expect(getBoundingClientRect).toHaveBeenCalledTimes(1);
  });
});
