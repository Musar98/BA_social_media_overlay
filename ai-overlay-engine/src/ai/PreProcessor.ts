class PreProcessor {
  private inputBuffer: Float32Array | null = null;
  private inputTensor: any | null = null;

  private lastW = 0;
  private lastH = 0;

  private resizeBuffers(w: number, h: number) {
    const HW = w * h;

    this.inputBuffer = new Float32Array(3 * HW);

    const ort = (self as any).ort;
    this.inputTensor = new ort.Tensor("float32", this.inputBuffer, [
      1,
      3,
      h,
      w,
    ]);

    this.lastW = w;
    this.lastH = h;
  }

  public prepareInput(width: number, height: number, img: Uint8ClampedArray) {
    if (width !== this.lastW || height !== this.lastH || !this.inputBuffer) {
      this.resizeBuffers(width, height);
    }

    if (!this.inputBuffer || !this.inputTensor) {
      throw new Error("Buffers not initialized");
    }

    const HW = this.lastW * this.lastH;

    let r = 0,
      g = HW,
      b = 2 * HW;
    const inv255 = 1 / 255;

    for (let i = 0; i < img.length; i += 4) {
      this.inputBuffer[r++] = img[i] * inv255;
      this.inputBuffer[g++] = img[i + 1] * inv255;
      this.inputBuffer[b++] = img[i + 2] * inv255;
    }

    return this.inputTensor;
  }
}

export const preProcessor = new PreProcessor();
