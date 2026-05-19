import { onnxRuntime } from "../ai/ONNXRuntime";
import { runPrediction } from "../ai/AIPredictor";
import { overlayLogger } from "../diagnostics/logger";

class PredictionTask {
  private canvas: OffscreenCanvas | null = null;
  private ctx: OffscreenCanvasRenderingContext2D | null = null;

  async initialize(): Promise<void> {
    await onnxRuntime.init();
  }

  async run(bitmap: ImageBitmap, generation?: number): Promise<void> {
    const t0 = performance.now();

    try {
      await onnxRuntime.init();
      const tInit = performance.now();

      if (!this.canvas || this.canvas.width !== bitmap.width || this.canvas.height !== bitmap.height) {
        this.canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
      }

      if (!this.ctx) {
        throw new Error("Failed to get 2D context from OffscreenCanvas");
      }

      this.ctx.drawImage(bitmap, 0, 0);
      const tDraw = performance.now();

      const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height).data;
      const tRead = performance.now();

      const aiParams = await runPrediction(
        imgData,
        this.canvas.width,
        this.canvas.height,
      );

      // Calculate image mean for contrast adjustment (Kornia style)
      let sum = 0;
      for (let i = 0; i < imgData.length; i += 4) {
        // weights: 0.299*R + 0.587*G + 0.114*B
        sum += (0.299 * imgData[i] + 0.587 * imgData[i + 1] + 0.114 * imgData[i + 2]) / 255.0;
      }
      aiParams.imageMean = sum / (imgData.length / 4);

      const tInference = performance.now();

      // timing log for performance measurment
      overlayLogger.verbose("ai-timing", {
        init: tInit - t0,
        draw: tDraw - tInit,
        readPixels: tRead - tDraw,
        inference: tInference - tRead,
        total: tInference - t0,
      });

      self.postMessage({ type: "prediction", aiParams, generation });
    } catch (err) {
      console.error("Worker error:", err);
      self.postMessage({ type: "error", error: String(err), generation });
    } finally {
      bitmap.close();
    }
  }
}

export const predictionTask = new PredictionTask();
