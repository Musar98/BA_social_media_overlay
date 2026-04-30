import { describe, it, expect, vi, beforeEach } from "vitest";
import { ONNXSessionState } from "../../src/state/state";
import { runPrediction } from "../../src/ai/runPrediction";
import { preProcessor } from "../../src/ai/PreProcessor";
import { postProcessor } from "../../src/ai/PostProcessor";

vi.mock("../../src/ai/PreProcessor", () => ({
  preProcessor: {
    prepareInput: vi.fn(),
  },
}));

vi.mock("../../src/ai/PostProcessor", () => ({
  postProcessor: {
    mapTensorToParams: vi.fn(),
  },
}));

describe("runPrediction", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    ONNXSessionState.session = undefined as any;
    ONNXSessionState.alphasTensor = undefined as any;
  });

  it("throws if alphasTensor is missing", async () => {
    ONNXSessionState.session = {} as any;

    await expect(runPrediction(new Uint8ClampedArray(), 1, 1)).rejects.toThrow(
      "ONNX alphas tensor missing for prediction",
    );
  });

  it("throws if session is missing", async () => {
    ONNXSessionState.alphasTensor = {} as any;

    await expect(runPrediction(new Uint8ClampedArray(), 1, 1)).rejects.toThrow(
      "ONNX session not initialized",
    );
  });

  it("runs full pipeline correctly", async () => {
    const fakeInput = { tensor: "input" };
    const fakeOutput = {
      transform_params: {
        data: new Float32Array([1, 2, 3]),
      },
    };
    const fakeResult = { some: "params" };

    (preProcessor.prepareInput as any).mockReturnValue(fakeInput);

    ONNXSessionState.alphasTensor = { alpha: true } as any;
    ONNXSessionState.session = {
      run: vi.fn().mockResolvedValue(fakeOutput),
    } as any;

    (postProcessor.mapTensorToParams as any).mockReturnValue(fakeResult);

    const pixels = new Uint8ClampedArray([1, 2, 3, 255]);

    const result = await runPrediction(pixels, 1, 1);

    expect(preProcessor.prepareInput).toHaveBeenCalledWith(1, 1, pixels);

    expect(ONNXSessionState?.session?.run).toHaveBeenCalledWith({
      images: fakeInput,
      alphas: ONNXSessionState.alphasTensor,
    });

    expect(postProcessor.mapTensorToParams).toHaveBeenCalledWith(
      fakeOutput.transform_params.data,
    );

    expect(result).toBe(fakeResult);
  });

  it("rethrows errors from session.run", async () => {
    const error = new Error("ONNX failed");

    (preProcessor.prepareInput as any).mockReturnValue({});

    ONNXSessionState.alphasTensor = {} as any;
    ONNXSessionState.session = {
      run: vi.fn().mockRejectedValue(error),
    } as any;

    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    await expect(runPrediction(new Uint8ClampedArray(), 1, 1)).rejects.toThrow(
      error,
    );

    expect(consoleSpy).toHaveBeenCalled();

    consoleSpy.mockRestore();
  });
});
