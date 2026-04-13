import { runPredictionTask } from "./runPredictionTask";

self.onmessage = async (e: MessageEvent) => {
  const { bitmap } = e.data;

  await runPredictionTask(bitmap);
};
