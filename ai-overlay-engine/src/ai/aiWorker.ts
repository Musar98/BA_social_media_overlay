import { initORTWorker } from "./onnxWorker";
import { runAIPredictionWorker } from "./runAiPredictionWorker"; // worker-safe AI prediction

self.onmessage = async (e: MessageEvent) => {
  const { bitmap } = e.data;

  try {
    await initORTWorker();

    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Failed to get 2D context from OffscreenCanvas");
    }

    ctx.drawImage(bitmap, 0, 0);
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

    const aiParams = await runAIPredictionWorker(
      imgData,
      canvas.width,
      canvas.height,
    );

    console.log("Worker finished AI prediction");
    self.postMessage({ aiParams });
  } catch (err) {
    console.error("Worker error:", err);
    self.postMessage({ error: String(err) });
  }
};
