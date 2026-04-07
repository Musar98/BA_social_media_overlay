import { createButton } from "./ui/button";
import { initVideoOverlay } from "./video/video";

async function main() {
  createButton();
  initVideoOverlay();
}

main()
  .then((r) => r)
  .catch((e) => console.error(e));
