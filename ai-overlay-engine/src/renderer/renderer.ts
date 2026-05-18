import { ImageTransformRenderer } from "../transformation/transformations";
import { CLIP_TO_UV } from "../transformation/ClipToUv";
import { IMAGE_ADJUSTMENTS_PIPELINE } from "../transformation/ImageAdjustmentsPipeline";

class Renderer {
  private instance: ImageTransformRenderer | null = null;
  private createCount = 0;
  private destroyCount = 0;
  private clearSourceCount = 0;

  init(canvas: HTMLCanvasElement): void {
    if (this.instance) {
      this.destroy();
    }

    this.createCount += 1;
    console.info("[ai-overlay] renderer created", {
      createCount: this.createCount,
    });

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

  clearSourceTexture(): void {
    if (!this.instance) return;

    this.clearSourceCount += 1;
    console.info("[ai-overlay] renderer source texture cleared", {
      clearSourceCount: this.clearSourceCount,
    });

    this.instance.clearSourceTexture();
  }

  destroy(): void {
    if (!this.instance) return;

    this.destroyCount += 1;
    console.info("[ai-overlay] renderer destroyed", {
      destroyCount: this.destroyCount,
    });

    this.instance.destroy();
    this.instance = null;
  }
}

export const renderer = new Renderer();
