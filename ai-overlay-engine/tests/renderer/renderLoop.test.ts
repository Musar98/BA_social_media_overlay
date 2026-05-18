// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderLoop } from "../../src/renderer/renderLoop";
import { UIState, AIState } from "../../src/state/state";
import { renderer } from "../../src/renderer/renderer";

vi.mock("../../src/renderer/renderer", () => ({
  renderer: {
    init: vi.fn(),
    renderFrame: vi.fn(),
    clearSourceTexture: vi.fn(),
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
    renderLoop.stop();
  });

  it("calls renderer.renderFrame when filter is enabled", async () => {
    UIState.filterEnabled = true;
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

  it("stops loop when video is removed from DOM", async () => {
    UIState.filterEnabled = true;
    renderLoop.start(video, canvas);

    document.body.removeChild(video);

    await new Promise(resolve => setTimeout(resolve, 10));

    expect(renderer.destroy).toHaveBeenCalled();
  });

  it("clears the renderer source texture before destroy on stop", () => {
    renderLoop.start(video, canvas);
    renderLoop.stop();

    expect(renderer.clearSourceTexture).toHaveBeenCalled();
    expect(renderer.destroy).toHaveBeenCalled();
  });

  it("terminates the AI worker on stop", () => {
    renderLoop.start(video, canvas);
    const worker = mockWorkers[0];

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
