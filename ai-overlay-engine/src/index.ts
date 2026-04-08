import { createButton } from "./ui/button";
import { observeVideos } from "./video/videoObserver";

async function main() {
  createButton();
  observeVideos();
}

main().catch(console.error);