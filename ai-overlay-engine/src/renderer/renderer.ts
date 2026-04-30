import { ImageTransformRenderer } from "../transformation/transformations";
import { CLIP_TO_UV } from "../transformation/ClipToUv";
import { IMAGE_ADJUSTMENTS_PIPELINE } from "../transformation/ImageAdjustmentsPipeline";

class Renderer {
  private instance: ImageTransformRenderer | null = null;

  init(canvas: HTMLCanvasElement): void {
    if (this.instance) {
      this.destroy();
    }

    this.instance = new ImageTransformRenderer(
      canvas,
      CLIP_TO_UV,
      IMAGE_ADJUSTMENTS_PIPELINE,
      { metrics: { enabled: false } },
    );
  }

  renderFrame(source: TexImageSource, params: any): void {
    this.instance?.renderFrame(source, params);
  }

  destroy(): void {
    if (!this.instance) return;
    this.instance.destroy();
    this.instance = null;
  }
}

export const renderer = new Renderer();
