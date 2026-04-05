import { observeVideos } from "./videoObserver";
import { modifyVideo } from "./videoModifier";

export function initVideoOverlay() {
  observeVideos(modifyVideo);
}
