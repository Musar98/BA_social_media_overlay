import { predictionTask } from "./runPredictionTask";

class AIWorkerHandler {
  constructor() {
    self.onmessage = async (e: MessageEvent) => {
      const { type, bitmap } = e.data;

      if (type === "init") {
        await predictionTask.warmup();
        (self as any).postMessage({ type: "initialized" });
        return;
      }

      if (bitmap) {
        await predictionTask.run(bitmap);
      }
    };
  }
}

new AIWorkerHandler();
