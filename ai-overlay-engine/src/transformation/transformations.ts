import { COLOR_CURVE_STEPS, MAX_PENDING_GPU_QUERIES, TONE_CURVE_STEPS } from "./Constants";

interface ImageTransformMetrics {
  type: "frame" | "gpu";
  frameId: number;
  fps: number | null;
  cpuFrameMs: number | null;
  cpuDrawMs: number | null;
  gpuFrameMs: number | null;
  gpuSupported: boolean;
  gpuDisjoint: boolean;
  timestampMs: number;
}

interface RendererOptions {
  metrics?: {
    enabled?: boolean;
    callback?: (metrics: ImageTransformMetrics) => void;
    fpsSmoothingWindow?: number;
  };
}

export class ImageTransformRenderer {
  public readonly canvas: HTMLCanvasElement;
  private readonly gl: WebGL2RenderingContext;
  private readonly program: WebGLProgram;
  private readonly vao: WebGLVertexArrayObject;
  private readonly vbo: WebGLBuffer;
  private readonly vertexCount: number;
  private readonly uniforms: Record<string, WebGLUniformLocation | null>;
  private readonly texture: WebGLTexture;
  private readonly toneCurveBuffer: Float32Array;
  private readonly colorCurveBuffer: Float32Array;

  private readonly metricsEnabled: boolean;
  private readonly metricsCallback: ((m: ImageTransformMetrics) => void) | null;
  private readonly fpsSmoothingWindow: number;
  private readonly timerExtension: any;
  private lastFrameTimestampMs = 0;
  private recentFrameDurationsMs: number[] = [];
  private pendingGpuQueries: Array<{ query: WebGLQuery; frameId: number }> = [];
  private frameCounter = 0;
  private lastCompletedGpuFrameMs: number | null = null;
  private lastCpuFrameMs: number | null = null;
  private lastCpuDrawMs: number | null = null;

  constructor(
    targetCanvas: HTMLCanvasElement,
    vertexShaderSource: string,
    fragmentShaderSource: string,
    options?: RendererOptions,
  ) {
    const gl = targetCanvas.getContext("webgl2", {
      alpha: false,
      depth: false,
      stencil: false,
      antialias: false,
      preserveDrawingBuffer: false,
      premultipliedAlpha: true,
      powerPreference: "high-performance",
    });

    if (!gl) {
      throw new Error("WebGL2 is not available for the target canvas");
    }

    this.canvas = targetCanvas;
    this.gl = gl;
    this.program = this.compileProgram(vertexShaderSource, fragmentShaderSource);
    const quad = this.createFullscreenQuad();
    this.vao = quad.vao;
    this.vbo = quad.vbo;
    this.vertexCount = quad.vertexCount;
    this.uniforms = this.resolveUniformLocations();
    this.texture = this.createSourceTexture();
    this.toneCurveBuffer = new Float32Array(TONE_CURVE_STEPS);
    this.colorCurveBuffer = new Float32Array(COLOR_CURVE_STEPS * 3);

    const metricsOptions = options?.metrics;
    this.metricsEnabled = Boolean(metricsOptions?.enabled);
    this.metricsCallback =
      typeof metricsOptions?.callback === "function"
        ? metricsOptions.callback
        : null;
    this.fpsSmoothingWindow = Math.max(
      1,
      metricsOptions?.fpsSmoothingWindow ?? 30,
    );
    this.timerExtension = this.metricsEnabled
      ? gl.getExtension("EXT_disjoint_timer_query_webgl2")
      : null;

    gl.useProgram(this.program);
    if (this.uniforms.image) {
      gl.uniform1i(this.uniforms.image, 0);
    }
  }

  // ---------------------------------------------------------------------------
  // Private – shader compilation
  // ---------------------------------------------------------------------------

  private compileShader(shaderType: number, shaderSource: string): WebGLShader {
    const shader = this.gl.createShader(shaderType);
    if (!shader) {
      throw new Error("Failed to create shader object");
    }
    this.gl.shaderSource(shader, shaderSource);
    this.gl.compileShader(shader);
    const isCompiled = this.gl.getShaderParameter(
      shader,
      this.gl.COMPILE_STATUS,
    );
    if (!isCompiled) {
      const compileLog =
        this.gl.getShaderInfoLog(shader) || "Unknown shader compile nvm error";
      this.gl.deleteShader(shader);
      throw new Error(
        `Shader compilation failed.\n\n${compileLog}\n\nSource:\n${shaderSource}`,
      );
    }
    return shader;
  }

  private compileProgram(
    vertexShaderSource: string,
    fragmentShaderSource: string,
  ): WebGLProgram {
    const vertexShader = this.compileShader(
      this.gl.VERTEX_SHADER,
      vertexShaderSource,
    );
    const fragmentShader = this.compileShader(
      this.gl.FRAGMENT_SHADER,
      fragmentShaderSource,
    );

    const program = this.gl.createProgram();
    if (!program) {
      this.gl.deleteShader(vertexShader);
      this.gl.deleteShader(fragmentShader);
      throw new Error("Failed to create WebGL program");
    }

    this.gl.attachShader(program, vertexShader);
    this.gl.attachShader(program, fragmentShader);
    this.gl.linkProgram(program);

    const isLinked = this.gl.getProgramParameter(program, this.gl.LINK_STATUS);
    this.gl.deleteShader(vertexShader);
    this.gl.deleteShader(fragmentShader);

    if (!isLinked) {
      const linkLog =
        this.gl.getProgramInfoLog(program) || "Unknown program link error";
      this.gl.deleteProgram(program);
      throw new Error(`Program linking failed.\n\n${linkLog}`);
    }

    return program;
  }

  // ---------------------------------------------------------------------------
  // Private – WebGL resource setup
  // ---------------------------------------------------------------------------

  private createFullscreenQuad(): {
    vao: WebGLVertexArrayObject;
    vbo: WebGLBuffer;
    vertexCount: number;
  } {
    const quadVertices = new Float32Array([
      -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1,
    ]);

    const vao = this.gl.createVertexArray();
    if (!vao) {
      throw new Error("Failed to create vertex array object");
    }

    const vbo = this.gl.createBuffer();
    if (!vbo) {
      this.gl.deleteVertexArray(vao);
      throw new Error("Failed to create vertex buffer object");
    }

    this.gl.bindVertexArray(vao);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vbo);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, quadVertices, this.gl.STATIC_DRAW);

    const positionAttributeLocation = this.gl.getAttribLocation(
      this.program,
      "a_pos",
    );
    if (positionAttributeLocation < 0) {
      this.gl.deleteBuffer(vbo);
      this.gl.deleteVertexArray(vao);
      throw new Error('Shader attribute "a_pos" was not found');
    }

    this.gl.enableVertexAttribArray(positionAttributeLocation);
    this.gl.vertexAttribPointer(
      positionAttributeLocation,
      2,
      this.gl.FLOAT,
      false,
      0,
      0,
    );

    this.gl.bindVertexArray(null);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);

    return { vao, vbo, vertexCount: 6 };
  }

  private resolveUniformLocations(): Record<
    string,
    WebGLUniformLocation | null
  > {
    return {
      image: this.gl.getUniformLocation(this.program, "u_image"),
      texelSize: this.gl.getUniformLocation(this.program, "u_texelSize"),
      canvasSize: this.gl.getUniformLocation(this.program, "u_canvasSize"),
      imageSize: this.gl.getUniformLocation(this.program, "u_imageSize"),
      sharp: this.gl.getUniformLocation(this.program, "u_sharp"),
      exposure: this.gl.getUniformLocation(this.program, "u_exposure"),
      contrast: this.gl.getUniformLocation(this.program, "u_contrast"),
      saturation: this.gl.getUniformLocation(this.program, "u_saturation"),
      blur: this.gl.getUniformLocation(this.program, "u_blur"),
      toneCurve: this.gl.getUniformLocation(this.program, "u_toneCurve"),
      colorCurve: this.gl.getUniformLocation(this.program, "u_colorCurve"),
    };
  }

  private createSourceTexture(): WebGLTexture {
    const texture = this.gl.createTexture();
    if (!texture) {
      throw new Error("Failed to create texture");
    }
    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_WRAP_S,
      this.gl.CLAMP_TO_EDGE,
    );
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_WRAP_T,
      this.gl.CLAMP_TO_EDGE,
    );
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_MIN_FILTER,
      this.gl.LINEAR,
    );
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_MAG_FILTER,
      this.gl.LINEAR,
    );
    this.gl.bindTexture(this.gl.TEXTURE_2D, null);
    return texture;
  }

  // ---------------------------------------------------------------------------
  // Private – metrics
  // ---------------------------------------------------------------------------

  private static nowMs(): number {
    if (
      typeof performance !== "undefined" &&
      typeof performance.now === "function"
    ) {
      return performance.now();
    }
    return Date.now();
  }

  private static computeAverageFps(
    frameDurationsMs: number[],
  ): number | null {
    if (frameDurationsMs.length === 0) return null;
    let sum = 0;
    for (let i = 0; i < frameDurationsMs.length; i++) {
      sum += frameDurationsMs[i];
    }
    if (sum <= 0) return null;
    return 1000 / (sum / frameDurationsMs.length);
  }

  private emitMetrics(metrics: ImageTransformMetrics): void {
    if (!this.metricsEnabled || !this.metricsCallback) return;
    this.metricsCallback(metrics);
  }

  private beginGpuTimerQuery(): {
    query: WebGLQuery | null;
    didBegin: boolean;
  } {
    if (!this.metricsEnabled || !this.timerExtension) {
      return { query: null, didBegin: false };
    }
    if (this.pendingGpuQueries.length >= MAX_PENDING_GPU_QUERIES) {
      return { query: null, didBegin: false };
    }
    const query = this.gl.createQuery();
    if (!query) {
      return { query: null, didBegin: false };
    }
    this.gl.beginQuery(this.timerExtension.TIME_ELAPSED_EXT, query);
    return { query, didBegin: true };
  }

  private endGpuTimerQuery(
    query: WebGLQuery | null,
    didBegin: boolean,
    frameId: number,
  ): void {
    if (!didBegin || !query || !this.timerExtension) return;
    this.gl.endQuery(this.timerExtension.TIME_ELAPSED_EXT);
    this.pendingGpuQueries.push({ query, frameId });
  }

  private pollCompletedGpuMetrics(): void {
    if (!this.metricsEnabled || !this.timerExtension) return;
    const ext = this.timerExtension;
    const disjoint = this.gl.getParameter(ext.GPU_DISJOINT_EXT);

    for (let i = 0; i < this.pendingGpuQueries.length; ) {
      const pending = this.pendingGpuQueries[i];
      const isAvailable = this.gl.getQueryParameter(
        pending.query,
        this.gl.QUERY_RESULT_AVAILABLE,
      );

      if (!isAvailable) {
        i += 1;
        continue;
      }

      let gpuFrameMs: number | null = null;
      if (!disjoint) {
        const elapsedNanoseconds = this.gl.getQueryParameter(
          pending.query,
          this.gl.QUERY_RESULT,
        );
        gpuFrameMs = elapsedNanoseconds / 1e6;
        this.lastCompletedGpuFrameMs = gpuFrameMs;
      }

      this.gl.deleteQuery(pending.query);
      this.pendingGpuQueries.splice(i, 1);

      this.emitMetrics({
        type: "gpu",
        frameId: pending.frameId,
        fps: ImageTransformRenderer.computeAverageFps(
          this.recentFrameDurationsMs,
        ),
        cpuFrameMs: this.lastCpuFrameMs,
        cpuDrawMs: this.lastCpuDrawMs,
        gpuFrameMs,
        gpuSupported: true,
        gpuDisjoint: Boolean(disjoint),
        timestampMs: ImageTransformRenderer.nowMs(),
      });
    }
  }

  private recordCpuMetrics(
    frameId: number,
    frameStartMs: number,
    drawStartMs: number,
    drawEndMs: number,
    frameEndMs: number,
  ): void {
    if (!this.metricsEnabled) return;

    const cpuDrawMs = drawEndMs - drawStartMs;
    const cpuFrameMs = frameEndMs - frameStartMs;

    this.lastCpuDrawMs = cpuDrawMs;
    this.lastCpuFrameMs = cpuFrameMs;

    if (this.lastFrameTimestampMs > 0) {
      const interFrameMs = frameEndMs - this.lastFrameTimestampMs;
      if (interFrameMs > 0) {
        this.recentFrameDurationsMs.push(interFrameMs);
        if (this.recentFrameDurationsMs.length > this.fpsSmoothingWindow) {
          this.recentFrameDurationsMs.shift();
        }
      }
    }

    this.lastFrameTimestampMs = frameEndMs;

    this.emitMetrics({
      type: "frame",
      frameId,
      fps: ImageTransformRenderer.computeAverageFps(
        this.recentFrameDurationsMs,
      ),
      cpuFrameMs,
      cpuDrawMs,
      gpuFrameMs: this.lastCompletedGpuFrameMs,
      gpuSupported: Boolean(this.timerExtension),
      gpuDisjoint: false,
      timestampMs: frameEndMs,
    });
  }

  // ---------------------------------------------------------------------------
  // Private – per-frame helpers
  // ---------------------------------------------------------------------------

  private static getSourceDimensions(
    source: TexImageSource,
  ): { width: number; height: number } {
    if ("videoWidth" in source && "videoHeight" in source) {
      return {
        width: Math.max(1, (source as HTMLVideoElement).videoWidth),
        height: Math.max(1, (source as HTMLVideoElement).videoHeight),
      };
    }
    if ("naturalWidth" in source && "naturalHeight" in source) {
      return {
        width: Math.max(1, (source as HTMLImageElement).naturalWidth),
        height: Math.max(1, (source as HTMLImageElement).naturalHeight),
      };
    }
    if ("width" in source && "height" in source) {
      return {
        width: Math.max(1, (source as HTMLCanvasElement).width),
        height: Math.max(1, (source as HTMLCanvasElement).height),
      };
    }
    throw new Error(
      "Unsupported source type: could not determine source dimensions",
    );
  }

  private static isRenderableSourceReady(source: TexImageSource): boolean {
    if (
      "readyState" in source &&
      "videoWidth" in source &&
      "videoHeight" in source
    ) {
      const video = source as HTMLVideoElement;
      return (
        video.readyState >= 2 &&
        video.videoWidth > 0 &&
        video.videoHeight > 0
      );
    }
    const { width, height } =
      ImageTransformRenderer.getSourceDimensions(source);
    return width > 0 && height > 0;
  }

  private clearCanvas(): void {
    this.gl.clearColor(0, 0, 0, 1);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);
  }

  private uploadSourceToTexture(source: TexImageSource): void {
    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture);
    this.gl.pixelStorei(this.gl.UNPACK_FLIP_Y_WEBGL, false);
    this.gl.texImage2D(
      this.gl.TEXTURE_2D,
      0,
      this.gl.RGBA,
      this.gl.RGBA,
      this.gl.UNSIGNED_BYTE,
      source,
    );
  }

  private static writeToneCurveToBuffer(
    toneCurve: number[] | Float32Array | undefined,
    targetBuffer: Float32Array,
  ): Float32Array {
    for (let i = 0; i < TONE_CURVE_STEPS; i++) {
      targetBuffer[i] = toneCurve?.[i] ?? 1.0;
    }
    return targetBuffer;
  }

  private static writeColorCurveToBuffer(
    colorCurve:
      | number[]
      | Float32Array
      | Array<[number, number, number]>
      | undefined,
    targetBuffer: Float32Array,
  ): Float32Array {
    if (!colorCurve) {
      for (let i = 0; i < targetBuffer.length; i++) {
        targetBuffer[i] = 1.0;
      }
      return targetBuffer;
    }
    if (Array.isArray(colorCurve) && Array.isArray(colorCurve[0])) {
      for (let i = 0; i < COLOR_CURVE_STEPS; i++) {
        const rgb = (colorCurve[i] as [number, number, number]) || [1, 1, 1];
        targetBuffer[i * 3] = rgb[0] ?? 1.0;
        targetBuffer[i * 3 + 1] = rgb[1] ?? 1.0;
        targetBuffer[i * 3 + 2] = rgb[2] ?? 1.0;
      }
      return targetBuffer;
    }
    for (let i = 0; i < targetBuffer.length; i++) {
      targetBuffer[i] = (colorCurve as number[])[i] ?? 1.0;
    }
    return targetBuffer;
  }

  private normalizeParams(params: any): {
    sharp: number;
    exposure: number;
    contrast: number;
    saturation: number;
    blur: number;
    toneCurve: Float32Array;
    colorCurve: Float32Array;
  } {
    const safeParams = params || {};
    return {
      sharp: safeParams.sharp ?? 0.0,
      exposure: safeParams.exposure ?? 0.0,
      contrast: safeParams.contrast ?? 1.0,
      saturation: safeParams.saturation ?? 1.0,
      blur: safeParams.blur ?? 0.0,
      toneCurve: ImageTransformRenderer.writeToneCurveToBuffer(
        safeParams.toneCurve,
        this.toneCurveBuffer,
      ),
      colorCurve: ImageTransformRenderer.writeColorCurveToBuffer(
        safeParams.colorCurve,
        this.colorCurveBuffer,
      ),
    };
  }

  private applyUniforms(
    sourceWidth: number,
    sourceHeight: number,
    params: {
      sharp: number;
      exposure: number;
      contrast: number;
      saturation: number;
      blur: number;
      toneCurve: Float32Array;
      colorCurve: Float32Array;
    },
  ): void {
    const { uniforms, gl, canvas } = this;

    if (uniforms.texelSize) {
      gl.uniform2f(uniforms.texelSize, 1 / sourceWidth, 1 / sourceHeight);
    }
    if (uniforms.canvasSize) {
      gl.uniform2f(uniforms.canvasSize, canvas.width, canvas.height);
    }
    if (uniforms.imageSize) {
      gl.uniform2f(uniforms.imageSize, sourceWidth, sourceHeight);
    }
    if (uniforms.sharp) gl.uniform1f(uniforms.sharp, params.sharp);
    if (uniforms.exposure) gl.uniform1f(uniforms.exposure, params.exposure);
    if (uniforms.contrast) gl.uniform1f(uniforms.contrast, params.contrast);
    if (uniforms.saturation) {
      gl.uniform1f(uniforms.saturation, params.saturation);
    }
    if (uniforms.blur) gl.uniform1f(uniforms.blur, params.blur);
    if (uniforms.toneCurve) gl.uniform1fv(uniforms.toneCurve, params.toneCurve);
    if (uniforms.colorCurve) {
      gl.uniform3fv(uniforms.colorCurve, params.colorCurve);
    }
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  renderFrame(
    source: TexImageSource,
    params: any,
  ): { width: number; height: number } {
    this.pollCompletedGpuMetrics();

    if (!ImageTransformRenderer.isRenderableSourceReady(source)) {
      this.clearCanvas();
      return { width: 0, height: 0 };
    }

    const frameId = ++this.frameCounter;
    const frameStartMs = ImageTransformRenderer.nowMs();

    const { width: sourceWidth, height: sourceHeight } =
      ImageTransformRenderer.getSourceDimensions(source);
    const normalizedParams = this.normalizeParams(params);

    if (this.canvas.width !== sourceWidth) this.canvas.width = sourceWidth;
    if (this.canvas.height !== sourceHeight) this.canvas.height = sourceHeight;
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    this.clearCanvas();
    this.uploadSourceToTexture(source);

    this.gl.useProgram(this.program);
    this.gl.bindVertexArray(this.vao);

    this.applyUniforms(sourceWidth, sourceHeight, normalizedParams);

    const drawStartMs = ImageTransformRenderer.nowMs();
    const gpuQueryState = this.beginGpuTimerQuery();

    this.gl.drawArrays(this.gl.TRIANGLES, 0, this.vertexCount);

    this.endGpuTimerQuery(
      gpuQueryState.query,
      gpuQueryState.didBegin,
      frameId,
    );

    this.gl.flush();

    const drawEndMs = ImageTransformRenderer.nowMs();


    this.recordCpuMetrics(
      frameId,
      frameStartMs,
      drawStartMs,
      drawEndMs,
      drawEndMs,
    );

    return { width: sourceWidth, height: sourceHeight };
  }

  getMetrics(): {
    fps: number | null;
    cpuFrameMs: number | null;
    cpuDrawMs: number | null;
    gpuFrameMs: number | null;
    gpuSupported: boolean;
  } {
    return {
      fps: ImageTransformRenderer.computeAverageFps(
        this.recentFrameDurationsMs,
      ),
      cpuFrameMs: this.lastCpuFrameMs,
      cpuDrawMs: this.lastCpuDrawMs,
      gpuFrameMs: this.lastCompletedGpuFrameMs,
      gpuSupported: Boolean(this.timerExtension),
    };
  }

  destroy(): void {
    for (let i = 0; i < this.pendingGpuQueries.length; i++) {
      this.gl.deleteQuery(this.pendingGpuQueries[i].query);
    }
    this.pendingGpuQueries.length = 0;
    this.gl.deleteTexture(this.texture);
    this.gl.deleteBuffer(this.vbo);
    this.gl.deleteVertexArray(this.vao);
    this.gl.deleteProgram(this.program);

    const ext = this.gl.getExtension("WEBGL_lose_context");
    ext?.loseContext();
  }
}
