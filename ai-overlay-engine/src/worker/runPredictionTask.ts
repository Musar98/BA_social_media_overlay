import { onnxRuntime } from "../ai/ONNXRuntime";
import { runPrediction } from "../ai/AIPredictor";

class PredictionTask {
  async run(bitmap: any): Promise<void> {
    const t0 = performance.now();

    try {
      await onnxRuntime.init();
      const tInit = performance.now();

      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        throw new Error("Failed to get 2D context from OffscreenCanvas");
      }

      ctx.drawImage(bitmap, 0, 0);
      const tDraw = performance.now();

      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      const tRead = performance.now();

      const aiParams = await runPrediction(
        imgData,
        canvas.width,
        canvas.height,
      );
      const tInference = performance.now();

      // timing log for performance measurment
      console.info("AI timing", JSON.stringify({
        init: tInit - t0,
        draw: tDraw - tInit,
        readPixels: tRead - tDraw,
        inference: tInference - tRead,
        total: tInference - t0,
      }));

      self.postMessage({ aiParams });
    } catch (err) {
      console.error("Worker error:", err);
      self.postMessage({ error: String(err) });
    }
  }
}

export const predictionTask = new PredictionTask();
