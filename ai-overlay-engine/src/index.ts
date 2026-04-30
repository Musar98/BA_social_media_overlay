import { filterButton } from "./ui/button";
import { videoObserver } from "./video/videoObserver";

class App {
  async start(): Promise<void> {
    filterButton.create();
    videoObserver.observe();
  }
}

new App().start().catch(console.error);
