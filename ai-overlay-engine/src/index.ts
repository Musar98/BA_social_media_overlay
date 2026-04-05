import { initInstagramFilter } from "./instagramFilter";
import { initORT } from "./ai/onnx";

async function main() {
  await initORT();
  initInstagramFilter();
}

main()
  .then((r) => r)
  .catch((e) => console.error(e));
