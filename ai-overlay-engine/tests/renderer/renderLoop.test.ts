// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderLoop } from "../../src/renderer/renderLoop";
import { UIState, AIState } from "../../src/state/state";
import { renderer } from "../../src/renderer/renderer";
import { createIdentityAIParams } from "../../src/ai/IdentityParams";

vi.mock("../../src/renderer/renderer", () => ({
  renderer: {
    init: vi.fn(),
    renderFrame: vi.fn(),
    clearSourceTexture: vi.fn(),
    resetSmoothing: vi.fn(),
    destroy: vi.fn(),
  },
}));

// Mock Worker
const mockWorkers: MockWorker[] = [];
class MockWorker {
  onmessage: ((ev: MessageEvent) => any) | null = null;
  postMessage = vi.fn();
  terminate = vi.fn();

  constructor() {
    mockWorkers.push(this);
  }
}
// @ts-ignore
(global as any).Worker = MockWorker;

// Mock requestAnimationFrame
// @ts-ignore
(global as any).requestAnimationFrame = vi.fn((cb) => setTimeout(cb, 0));
// @ts-ignore
(global as any).cancelAnimationFrame = vi.fn();

describe("RenderLoop", () => {
  let video: HTMLVideoElement;
  let canvas: HTMLCanvasElement;

  beforeEach(() => {
    vi.clearAllMocks();
    mockWorkers.length = 0;
    video = document.createElement("video");
    canvas = document.createElement("canvas");
    document.body.appendChild(video);
    UIState.filterEnabled = false;
    AIState.params = undefined;
  });

  it("initializes renderer on start", () => {
    renderLoop.start(video, canvas);
    expect(renderer.init).toHaveBeenCalledWith(canvas);
    expect(renderer.resetSmoothing).toHaveBeenCalled();
    renderLoop.stop();
  });

  it("starts every video with identity AI params", () => {
    AIState.params = {
      sharp: 9,
      exposure: 9,
      contrast: 9,
      saturation: 9,
      blur: 9,
      imageMean: 9,
      toneCurve: [9],
      colorCurve: [9],
    };

    renderLoop.start(video, canvas);

    expect(AIState.params).toEqual(createIdentityAIParams());

    renderLoop.stop();
  });

  it("calls renderer.renderFrame when filter is enabled", async () => {
    UIState.filterEnabled = true;
    (renderer.renderFrame as any).mockReturnValue({ width: 100, height: 100 });
    // Mock readyState to indicate video is ready
    Object.defineProperty(video, 'readyState', { value: 4 });
    Object.defineProperty(video, 'paused', { value: false });
    Object.defineProperty(video, 'ended', { value: false });

    renderLoop.start(video, canvas);

    // Give it a tick to run the loop
    await new Promise(resolve => setTimeout(resolve, 10));

    expect(renderer.renderFrame).toHaveBeenCalled();
    renderLoop.stop();
  });

  it("shows the overlay immediately when filter is enabled", async () => {
    UIState.filterEnabled = true;
    (renderer.renderFrame as any).mockReturnValue({ width: 0, height: 0 });
    Object.defineProperty(video, "readyState", { value: 4 });
    Object.defineProperty(video, "paused", { value: false });
    Object.defineProperty(video, "ended", { value: false });

    renderLoop.start(video, canvas);

    await new Promise(resolve => setTimeout(resolve, 100));

    expect(canvas.style.display).toBe("block");
    expect(video.style.opacity).toBe("0");

    renderLoop.stop();
  });

  it("does not rewrite visibility styles after the overlay is already visible", async () => {
    UIState.filterEnabled = true;
    (renderer.renderFrame as any).mockReturnValue({ width: 100, height: 100 });
    Object.defineProperty(video, "readyState", { value: 4 });
    Object.defineProperty(video, "paused", { value: false });
    Object.defineProperty(video, "ended", { value: false });

    const displaySetter = vi.fn();
    const opacitySetter = vi.fn();

    Object.defineProperty(canvas.style, "display", {
      configurable: true,
      set: displaySetter,
      get: () => "block",
    });

    Object.defineProperty(video.style, "opacity", {
      configurable: true,
      set: opacitySetter,
      get: () => "0",
    });

    renderLoop.start(video, canvas);

    await new Promise(resolve => setTimeout(resolve, 100));

    expect(displaySetter).toHaveBeenCalledTimes(1);
    expect(displaySetter).toHaveBeenCalledWith("block");
    expect(opacitySetter).toHaveBeenCalledTimes(1);
    expect(opacitySetter).toHaveBeenCalledWith("0");

    renderLoop.stop();
  });

  it("keeps the native video visible when filter is disabled", () => {
    UIState.filterEnabled = false;

    renderLoop.start(video, canvas);

    expect(canvas.style.display).toBe("none");
    expect(video.style.opacity).toBe("1");

    renderLoop.stop();
  });

  it("stops loop when video is removed from DOM", async () => {
    UIState.filterEnabled = true;
    renderLoop.start(video, canvas);

    document.body.removeChild(video);

    await new Promise(resolve => setTimeout(resolve, 100));

    expect(renderer.destroy).toHaveBeenCalled();
  });

  it("clears the renderer source texture before destroy on stop", () => {
    renderLoop.start(video, canvas);
    renderLoop.stop();

    expect(renderer.clearSourceTexture).toHaveBeenCalled();
    expect(renderer.destroy).toHaveBeenCalled();
  });

  it("does not create the AI worker before the first valid rendered frame", async () => {
    UIState.filterEnabled = true;
    (renderer.renderFrame as any).mockReturnValue({ width: 0, height: 0 });
    Object.defineProperty(video, "readyState", { value: 4 });
    Object.defineProperty(video, "paused", { value: false });
    Object.defineProperty(video, "ended", { value: false });

    renderLoop.start(video, canvas);

    await new Promise(resolve => setTimeout(resolve, 50));

    expect(mockWorkers).toHaveLength(0);

    renderLoop.stop();
  });

  it("terminates the AI worker on stop after a valid rendered frame warms it", async () => {
    UIState.filterEnabled = true;
    (renderer.renderFrame as any).mockReturnValue({ width: 100, height: 100 });
    Object.defineProperty(video, "readyState", { value: 4 });
    Object.defineProperty(video, "paused", { value: false });
    Object.defineProperty(video, "ended", { value: false });

    renderLoop.start(video, canvas);

    await new Promise(resolve => setTimeout(resolve, 50));

    const worker = mockWorkers[0];
    expect(worker).toBeDefined();

    renderLoop.stop();

    expect(worker.terminate).toHaveBeenCalled();
  });

  it("uses requestVideoFrameCallback when available", async () => {
    UIState.filterEnabled = true;
    Object.defineProperty(video, "readyState", { value: 4 });
    Object.defineProperty(video, "paused", { value: false });
    Object.defineProperty(video, "ended", { value: false });

    const cancelVideoFrameCallback = vi.fn();
    let scheduled = false;
    const requestVideoFrameCallback = vi.fn((cb) => {
      if (!scheduled) {
        scheduled = true;
        setTimeout(cb, 0);
      }

      return 7;
    });

    Object.assign(video, {
      requestVideoFrameCallback,
      cancelVideoFrameCallback,
    });

    renderLoop.start(video, canvas);

    await new Promise(resolve => setTimeout(resolve, 10));

    expect(requestVideoFrameCallback).toHaveBeenCalled();
    expect(renderer.renderFrame).toHaveBeenCalled();

    renderLoop.stop();

    expect(cancelVideoFrameCallback).toHaveBeenCalledWith(7);
  });
});
