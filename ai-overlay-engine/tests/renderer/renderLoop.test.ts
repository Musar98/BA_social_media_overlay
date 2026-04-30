// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderLoop } from "../../src/renderer/renderLoop";
import { UIState, AIState } from "../../src/state/state";
import { renderer } from "../../src/renderer/renderer";

vi.mock("../../src/renderer/renderer", () => ({
  renderer: {
    init: vi.fn(),
    renderFrame: vi.fn(),
    destroy: vi.fn(),
  },
}));

// Mock Worker
class MockWorker {
  onmessage: ((ev: MessageEvent) => any) | null = null;
  postMessage = vi.fn();
  terminate = vi.fn();
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
});
