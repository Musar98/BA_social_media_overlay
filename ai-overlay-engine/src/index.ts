import { initORT } from "./ai/onnx";
import { createButton } from "./ui/button";
import { observeVideos } from "./video/videoObserver";

async function main() {
  await initORT();
  createButton();
  observeVideos();
}

main().catch(console.error);