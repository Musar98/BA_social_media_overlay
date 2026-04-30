import { predictionTask } from "./runPredictionTask";

class AIWorkerHandler {
  constructor() {
    self.onmessage = async (e: MessageEvent) => {
      const { bitmap } = e.data;
      await predictionTask.run(bitmap);
    };
  }
}

new AIWorkerHandler();
