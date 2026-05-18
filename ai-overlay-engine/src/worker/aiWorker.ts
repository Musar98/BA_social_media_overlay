import { predictionTask } from "./runPredictionTask";

class AIWorkerHandler {
  constructor() {
    self.onmessage = async (e: MessageEvent) => {
      const { generation, type, bitmap } = e.data;

      if (type === "init") {
        try {
          await predictionTask.initialize();
          (self as any).postMessage({ type: "initialized" });
        } catch (err) {
          (self as any).postMessage({ type: "error", error: String(err) });
        }
        return;
      }

      if (type === "predict" && bitmap) {
        await predictionTask.run(bitmap, generation);
      }
    };
  }
}

new AIWorkerHandler();
