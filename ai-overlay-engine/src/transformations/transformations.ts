const clipToUV = `#version 300 es
/*
Takes 2D positions (a_pos) in clip space (range -1 to 1)
Converts those positions into UV coordinates (0 → 1)
Flips the Y-axis so textures don’t appear upside down
Passes the UVs to the fragment shader
Outputs the vertex position unchanged to the screen
*/

// Input vertex position in clip space (-1 to +1)
in vec2 a_pos;

// UV coordinates passed to fragment shader (0 to 1)
out vec2 v_uv;

void main() {
  // Convert clip space (-1..1) → UV space (0..1)
  // x: map -1 → 0 and +1 → 1
  // y: map -1 → 1 and +1 → 0 (flip Y axis)
  v_uv = vec2(0.5f * (a_pos.x + 1.0f),      // scale and shift X
  1.0f - 0.5f * (a_pos.y + 1.0f) // scale, shift, then flip Y
  );

  // Output final position (no transformation)
  gl_Position = vec4(a_pos, 0.0f, 1.0f);
}`;
const imageAdjPipeline = `#version 300 es
precision highp float;

// Interpolated UV coordinates from the vertex shader.
// These are canvas-space UVs in the range [0, 1].
in vec2 v_uv;

// Final output color for this fragment.
out vec4 outColor;

// Source image texture.
uniform sampler2D u_image;

// Size of one texel in UV space, typically:
// vec2(1.0 / imageWidth, 1.0 / imageHeight)
uniform vec2 u_texelSize;

// Size of the output canvas in pixels.
uniform vec2 u_canvasSize;

// Size of the source image in pixels.
uniform vec2 u_imageSize;

// User-controlled adjustment parameters.
//uniform float u_gamma;       // Gamma adjustment
uniform float u_sharp;       // Edge-based sharpening strength
//uniform float u_wb;          // White balance blend factor
uniform float u_exposure;    // Exposure in stops
//uniform float u_bright;      // Linear brightness offset
uniform float u_contrast;    // Contrast multiplier around 0.5
uniform float u_saturation;  // Saturation multiplier
//uniform float u_bw;          // Black-and-white mix amount
//uniform float u_hue;         // Hue rotation in radians
uniform float u_blur;        // Blur radius multiplier

// Piecewise tone curve weights.
// Each segment influences a fixed tonal range of the input color.
uniform float u_toneCurve[8];

// Piecewise per-channel color curve weights.
// Each segment is a vec3 so RGB channels can be shaped independently.
uniform vec3 u_colorCurve[8];

// Computes perceptual luminance using weighted RGB components.
// This is used when blending toward grayscale.
float rgb2lum(vec3 c) {
    return 0.27f * c.r + 0.67f * c.g + 0.06f * c.b;
}

// Converts RGB to HSV.
// Hue is stored in [0, 1), saturation and value in [0, 1].
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10f;

    return vec3(abs(q.z + (q.w - q.y) / (6.0f * d + e)), d / (q.x + e), q.x);
}

// Converts HSV back to RGB.
vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0f, 2.0f / 3.0f, 1.0f / 3.0f)) * 6.0f - 3.0f);
    return c.z * mix(vec3(1.0f), clamp(p - 1.0f, 0.0f, 1.0f), c.y);
}

// Applies gamma using a power curve.
// Small clamps prevent invalid pow() behavior for near-zero values.
//vec3 applyGamma(vec3 c, float gammaValue) {
//    return pow(max(c, vec3(1e-7f)), vec3(max(gammaValue, 0.01f)));
//}

// Applies exposure in photographic stops.
// exp2(x) means +1.0 doubles brightness, -1.0 halves it.
vec3 applyExposure(vec3 c, float exposureValue) {
    return clamp(c * exp2(exposureValue), 0.0f, 1.0f);
}

// Applies a simple additive brightness offset.
//vec3 applyBrightness(vec3 c, float brightValue) {
//    return clamp(c + brightValue, 0.0f, 1.0f);
//}

// Applies contrast around a midpoint of 0.5.
// Values > 1.0 increase contrast, values < 1.0 reduce contrast.
vec3 applyContrast(vec3 c, float contrastValue) {
    return clamp((c - 0.5f) * contrastValue + 0.5f, 0.0f, 1.0f);
}

// Adjusts saturation in HSV space.
// Saturation is scaled, then converted back to RGB.
vec3 applySaturation(vec3 c, float satValue) {
    vec3 hsv = rgb2hsv(c);
    hsv.y = clamp(hsv.y * satValue, 0.0f, 1.0f);
    return clamp(hsv2rgb(hsv), 0.0f, 1.0f);
}

// Rotates hue by the given angle in radians.
// Hue is stored on a wrapped [0, 1) interval.
//vec3 applyHue(vec3 c, float hueRadians) {
//    vec3 hsv = rgb2hsv(c);
//    hsv.x = fract(hsv.x + hueRadians / (2.0f * 3.141592653589793f));
//    return clamp(hsv2rgb(hsv), 0.0f, 1.0f);
//}

// Blends the color toward grayscale based on luminance.
// bwValue = 0.0 keeps color, bwValue = 1.0 produces full grayscale.
//vec3 applyBlackWhite(vec3 c, float bwValue) {
//    float l = rgb2lum(c);
//    return mix(c, vec3(l), clamp(bwValue, 0.0f, 1.0f));
//}

// Applies a simple channel rebalance intended as white balance.
// This is not a standard temperature/tint model.
// It computes a per-pixel balancing factor and blends toward it.
//vec3 applyWhiteBalance(vec3 c, float wbValue) {
//    vec3 balancing = vec3(0.5f / max(c.r, 1e-6f), 0.5f / max(c.g, 1e-6f), 0.5f / max(c.b, 1e-6f));
//
//    float avg = (c.r + c.g + c.b) / 3.0f + 1e-6f;
//    vec3 balanced = clamp(c * balancing * avg, 0.0f, 1.0f);
//
//    return mix(c, balanced, clamp(wbValue, 0.0f, 1.0f));
//}

// Applies a piecewise tone curve across 8 equal input ranges.
// Each segment contributes proportionally within its range.
// This acts like a custom scalar tone remapping applied equally to RGB.
vec3 applyToneCurve(vec3 c) {
    const int STEPS = 8;
    vec3 total = vec3(0.0f);
    float stepsF = float(STEPS);

    for(int i = 0; i < STEPS; i++) {
        float fi = float(i);

    // Extract only the portion of c that falls into this segment.
        vec3 seg = clamp(c - fi / stepsF, 0.0f, 1.0f / stepsF);

    // Weight that segment and accumulate it.
        total += seg * u_toneCurve[i];
    }

    return clamp(total, 0.0f, 1.0f);
}

// Applies a piecewise per-channel color curve across 8 equal input ranges.
// Unlike the tone curve, each segment uses a vec3 weight so RGB can vary.
vec3 applyColorCurve(vec3 c) {
    const int STEPS = 8;
    vec3 total = vec3(0.0f);
    float stepsF = float(STEPS);

    for(int i = 0; i < STEPS; i++) {
        float fi = float(i);

    // Extract only the portion of c that falls into this segment.
        vec3 seg = clamp(c - fi / stepsF, 0.0f, 1.0f / stepsF);

    // Apply per-channel weighting.
        total += seg * u_colorCurve[i];
    }

    return clamp(total, 0.0f, 1.0f);
}

// Samples the source image and returns RGB only.
vec3 sampleImage(vec2 uv) {
    return texture(u_image, uv).rgb;
}

// Computes Sobel edge magnitude from neighboring pixels.
// This is used as an edge signal for sharpening / enhancement.
vec3 sobelMagnitude(vec2 uv, vec2 texel) {
    vec3 tl = sampleImage(uv + texel * vec2(-1.0f, -1.0f));
    vec3 tc = sampleImage(uv + texel * vec2(0.0f, -1.0f));
    vec3 tr = sampleImage(uv + texel * vec2(1.0f, -1.0f));
    vec3 ml = sampleImage(uv + texel * vec2(-1.0f, 0.0f));
    vec3 mr = sampleImage(uv + texel * vec2(1.0f, 0.0f));
    vec3 bl = sampleImage(uv + texel * vec2(-1.0f, 1.0f));
    vec3 bc = sampleImage(uv + texel * vec2(0.0f, 1.0f));
    vec3 br = sampleImage(uv + texel * vec2(1.0f, 1.0f));

    vec3 gx = -tl - 2.0f * ml - bl + tr + 2.0f * mr + br;
    vec3 gy = -tl - 2.0f * tc - tr + bl + 2.0f * bc + br;

    return sqrt(gx * gx + gy * gy + 1e-7f);
}

// Applies edge-based sharpening.
// This is not classic unsharp masking. Instead, detected edge energy
// is multiplied back into the current color and boosted by sharpValue.
vec3 applySharpen(vec3 c, vec2 uv, vec2 texel, float sharpValue) {
    if(sharpValue <= 0.001f)
        return c;

    vec3 edges = sobelMagnitude(uv, texel);
    return clamp(c + sharpValue * edges * c, 0.0f, 1.0f);
}

// Converts canvas UV coordinates into image UV coordinates while preserving
// image aspect ratio inside the canvas.
//
// Returns false when the current fragment lies outside the displayed image area.
// In that case the caller can output a background color instead.
bool mapCanvasUVToImageUV(vec2 canvasUV, out vec2 imageUV) {
    float canvasAspect = u_canvasSize.x / u_canvasSize.y;
    float imageAspect = u_imageSize.x / u_imageSize.y;

    vec2 scale = vec2(1.0f);

  // Fit the image into the canvas while preserving aspect ratio.
  // One axis may be letterboxed.
    if(imageAspect > canvasAspect) {
        scale.y = canvasAspect / imageAspect;
    } else {
        scale.x = imageAspect / canvasAspect;
    }

  // Recenter and remap the canvas UV into image UV space.
    vec2 uv = (canvasUV - 0.5f) / scale + 0.5f;

  // Reject fragments outside the visible image region.
    if(uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        return false;
    }

    imageUV = uv;
    return true;
}

// Processes a single image sample through the full adjustment stack.
// The order matters and directly affects the final look.
vec3 processSample(vec2 uv) {
    vec3 color = sampleImage(uv);

    //color = applyGamma(color, u_gamma);
    color = applySharpen(color, uv, u_texelSize, u_sharp);
    //color = applyWhiteBalance(color, u_wb);
    color = applyExposure(color, u_exposure);
    //color = applyBrightness(color, u_bright);
    color = applyContrast(color, u_contrast);
    color = applySaturation(color, u_saturation);
    //color = applyBlackWhite(color, u_bw);
    color = applyToneCurve(color);
    //color = applyHue(color, u_hue);
    color = applyColorCurve(color);

    return clamp(color, 0.0f, 1.0f);
}

// Applies a small Gaussian-like blur by sampling a 5x5 neighborhood.
// Important: each tap is fully processed first, then averaged.
// That means blur happens over the edited result, not the raw image.
vec3 sampleProcessedBlur(vec2 uv, vec2 texel, float radius) {
    vec3 sum = vec3(0.0f);
    float wsum = 0.0f;

    for(int y = -2; y <= 2; y++) {
        for(int x = -2; x <= 2; x++) {
            vec2 off = vec2(float(x), float(y)) * texel * radius;
            float d2 = float(x * x + y * y);
            float w = exp(-d2 / 4.0f);

            sum += processSample(uv + off) * w;
            wsum += w;
        }
    }

    return sum / max(wsum, 1e-6f);
}

// Main fragment entry point.
void main() {
    vec2 imageUV;

  // Convert canvas-space UV to image-space UV.
  // Output black for pixels outside the fitted image area.
    if(!mapCanvasUVToImageUV(v_uv, imageUV)) {
        outColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    vec3 color;

  // Optional blur path.
    if(u_blur > 0.001f) {
        color = sampleProcessedBlur(imageUV, u_texelSize, u_blur);
    } else {
        color = processSample(imageUV);
    }

    outColor = vec4(clamp(color, 0.0f, 1.0f), 1.0f);
}`;

/**
 * Number of tone-curve segments expected by the shader.
 * This is fixed here because the fragment shader declares u_toneCurve[8].
 */
const TONE_CURVE_STEPS = 8;

/**
 * Number of color-curve segments expected by the shader.
 * This is fixed here because the fragment shader declares u_colorCurve[8].
 */
const COLOR_CURVE_STEPS = 8;

/**
 * Maximum number of pending GPU timer queries we keep alive at once.
 *
 * Why this exists:
 * GPU timer queries resolve asynchronously. A small cap prevents unbounded
 * growth if frames are rendered faster than results become available.
 */
const MAX_PENDING_GPU_QUERIES = 8;

/**
 * Compiles a single GLSL shader.
 *
 * Why this is separated:
 * shader compilation is one of the first failure points in a WebGL pipeline.
 * Keeping it isolated makes errors easier to diagnose and reuse simpler.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {number} shaderType
 * @param {string} shaderSource
 * @returns {WebGLShader}
 */
function compileGLSLShader(gl, shaderType, shaderSource) {
  const shader = gl.createShader(shaderType);

  if (!shader) {
    throw new Error("Failed to create shader object");
  }

  gl.shaderSource(shader, shaderSource);
  gl.compileShader(shader);

  const isCompiled = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
  if (!isCompiled) {
    const compileLog =
      gl.getShaderInfoLog(shader) || "Unknown shader compile error";
    gl.deleteShader(shader);

    throw new Error(
      `Shader compilation failed.\n\n${compileLog}\n\nSource:\n${shaderSource}`,
    );
  }

  return shader;
}

/**
 * Creates a linked program from a vertex and fragment shader.
 *
 * Why linking is wrapped in its own function:
 * the caller should not have to care about shader lifecycle details such as
 * attaching, linking, and deleting intermediate shader objects.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {string} vertexShaderSource
 * @param {string} fragmentShaderSource
 * @returns {WebGLProgram}
 */
function createGLSLProgram(gl, vertexShaderSource, fragmentShaderSource) {
  const vertexShader = compileGLSLShader(
    gl,
    gl.VERTEX_SHADER,
    vertexShaderSource,
  );
  const fragmentShader = compileGLSLShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource,
  );

  const program = gl.createProgram();
  if (!program) {
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    throw new Error("Failed to create WebGL program");
  }

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  const isLinked = gl.getProgramParameter(program, gl.LINK_STATUS);

  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);

  if (!isLinked) {
    const linkLog =
      gl.getProgramInfoLog(program) || "Unknown program link error";
    gl.deleteProgram(program);
    throw new Error(`Program linking failed.\n\n${linkLog}`);
  }

  return program;
}

/**
 * Creates a fullscreen quad VAO/VBO pair.
 *
 * Why a fullscreen quad is used:
 * image processing shaders do not need scene geometry. Instead, we draw a
 * rectangle that covers the full output canvas and let the fragment shader
 * run once per output pixel.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLProgram} program
 * @returns {{ vao: WebGLVertexArrayObject, vbo: WebGLBuffer, vertexCount: number }}
 */
function createFullscreenQuad(gl, program) {
  const quadVertices = new Float32Array([
    -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1,
  ]);

  const vao = gl.createVertexArray();
  if (!vao) {
    throw new Error("Failed to create vertex array object");
  }

  const vbo = gl.createBuffer();
  if (!vbo) {
    gl.deleteVertexArray(vao);
    throw new Error("Failed to create vertex buffer object");
  }

  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);

  const positionAttributeLocation = gl.getAttribLocation(program, "a_pos");
  if (positionAttributeLocation < 0) {
    gl.deleteBuffer(vbo);
    gl.deleteVertexArray(vao);
    throw new Error('Shader attribute "a_pos" was not found');
  }

  gl.enableVertexAttribArray(positionAttributeLocation);
  gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  return {
    vao,
    vbo,
    vertexCount: 6,
  };
}

/**
 * Resolves and caches all uniform locations used by the shader.
 *
 * Why locations are cached:
 * uniform lookups are relatively expensive and should not be repeated on
 * every render call.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLProgram} program
 * @returns {Record<string, WebGLUniformLocation | null>}
 */
function getImageTransformUniformLocations(gl, program) {
  return {
    image: gl.getUniformLocation(program, "u_image"),
    texelSize: gl.getUniformLocation(program, "u_texelSize"),
    canvasSize: gl.getUniformLocation(program, "u_canvasSize"),
    imageSize: gl.getUniformLocation(program, "u_imageSize"),
    sharp: gl.getUniformLocation(program, "u_sharp"),
    exposure: gl.getUniformLocation(program, "u_exposure"),
    contrast: gl.getUniformLocation(program, "u_contrast"),
    saturation: gl.getUniformLocation(program, "u_saturation"),
    blur: gl.getUniformLocation(program, "u_blur"),
    toneCurve: gl.getUniformLocation(program, "u_toneCurve"),
    colorCurve: gl.getUniformLocation(program, "u_colorCurve"),
  };
}

/**
 * Creates and configures the source texture.
 *
 * Why this texture setup is used:
 * - CLAMP_TO_EDGE avoids border artifacts during blur and filtered sampling
 * - LINEAR filtering gives smoother results when the source is sampled
 * - one dedicated texture object allows repeated uploads of new frames
 *
 * @param {WebGL2RenderingContext} gl
 * @returns {WebGLTexture}
 */
function createSourceTexture(gl) {
  const texture = gl.createTexture();

  if (!texture) {
    throw new Error("Failed to create texture");
  }

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  gl.bindTexture(gl.TEXTURE_2D, null);

  return texture;
}

/**
 * Creates a monotonic timestamp in milliseconds.
 *
 * Why this helper exists:
 * performance.now() is preferred in browsers, but Date.now() keeps this
 * code resilient in less typical environments.
 *
 * @returns {number}
 */
function nowMs() {
  if (
    typeof performance !== "undefined" &&
    typeof performance.now === "function"
  ) {
    return performance.now();
  }

  return Date.now();
}

/**
 * Creates the optional metrics subsystem.
 *
 * Supported metrics:
 * - fps
 * - cpuFrameMs
 * - cpuDrawMs
 * - gpuFrameMs when EXT_disjoint_timer_query_webgl2 is available
 *
 * The callback is invoked as metrics become available. GPU metrics can arrive
 * one or more frames later because timer queries resolve asynchronously.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {{
 *   enabled?: boolean,
 *   callback?: ((metrics: ImageTransformMetrics) => void) | undefined,
 *   fpsSmoothingWindow?: number | undefined
 * } | undefined} metricsOptions
 * @returns {ImageTransformMetricsState}
 */
function createRendererMetricsState(gl, metricsOptions) {
  const enabled = Boolean(metricsOptions?.enabled);
  const callback =
    typeof metricsOptions?.callback === "function"
      ? metricsOptions.callback
      : null;

  const fpsSmoothingWindow = Math.max(
    1,
    metricsOptions?.fpsSmoothingWindow ?? 30,
  );

  const timerExtension = enabled
    ? gl.getExtension("EXT_disjoint_timer_query_webgl2")
    : null;

  return {
    enabled,
    callback,
    fpsSmoothingWindow,
    timerExtension,
    lastFrameTimestampMs: 0,
    recentFrameDurationsMs: [],
    pendingGpuQueries: [],
    frameCounter: 0,
    lastCompletedGpuFrameMs: null,
    lastCpuFrameMs: null,
    lastCpuDrawMs: null,
  };
}

/**
 * Begins a GPU timer query when supported.
 *
 * Why this is separate:
 * GPU timing should stay completely optional and isolated from the normal
 * render path.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {ImageTransformMetricsState} metricsState
 * @returns {{ query: WebGLQuery | null, didBegin: boolean }}
 */
function beginGpuTimerQuery(gl, metricsState) {
  if (!metricsState.enabled || !metricsState.timerExtension) {
    return { query: null, didBegin: false };
  }

  if (metricsState.pendingGpuQueries.length >= MAX_PENDING_GPU_QUERIES) {
    return { query: null, didBegin: false };
  }

  const query = gl.createQuery();
  if (!query) {
    return { query: null, didBegin: false };
  }

  gl.beginQuery(metricsState.timerExtension.TIME_ELAPSED_EXT, query);

  return { query, didBegin: true };
}

/**
 * Ends a GPU timer query if one had been started.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {ImageTransformMetricsState} metricsState
 * @param {WebGLQuery | null} query
 * @param {boolean} didBegin
 * @param {number} frameId
 */
function endGpuTimerQuery(gl, metricsState, query, didBegin, frameId) {
  if (!didBegin || !query || !metricsState.timerExtension) {
    return;
  }

  gl.endQuery(metricsState.timerExtension.TIME_ELAPSED_EXT);

  metricsState.pendingGpuQueries.push({
    query,
    frameId,
  });
}

/**
 * Computes average FPS from recent frame durations.
 *
 * @param {number[]} frameDurationsMs
 * @returns {number | null}
 */
function computeAverageFps(frameDurationsMs) {
  if (frameDurationsMs.length === 0) {
    return null;
  }

  let sum = 0;
  for (let i = 0; i < frameDurationsMs.length; i++) {
    sum += frameDurationsMs[i];
  }

  if (sum <= 0) {
    return null;
  }

  return 1000 / (sum / frameDurationsMs.length);
}

/**
 * Emits metrics to the user callback if enabled.
 *
 * @param {ImageTransformMetricsState} metricsState
 * @param {ImageTransformMetrics} metrics
 */
function emitMetrics(metricsState, metrics) {
  if (!metricsState.enabled || !metricsState.callback) {
    return;
  }

  metricsState.callback(metrics);
}

/**
 * Polls pending GPU timer queries and emits any completed GPU metrics.
 *
 * Why polling is necessary:
 * GPU timing results are not immediately available after draw submission.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {ImageTransformMetricsState} metricsState
 */
function pollCompletedGpuMetrics(gl, metricsState) {
  if (!metricsState.enabled || !metricsState.timerExtension) {
    return;
  }

  const ext = metricsState.timerExtension;
  const disjoint = gl.getParameter(ext.GPU_DISJOINT_EXT);

  for (let i = 0; i < metricsState.pendingGpuQueries.length; ) {
    const pending = metricsState.pendingGpuQueries[i];
    const isAvailable = gl.getQueryParameter(
      pending.query,
      gl.QUERY_RESULT_AVAILABLE,
    );

    if (!isAvailable) {
      i += 1;
      continue;
    }

    let gpuFrameMs = null;

    if (!disjoint) {
      const elapsedNanoseconds = gl.getQueryParameter(
        pending.query,
        gl.QUERY_RESULT,
      );
      gpuFrameMs = elapsedNanoseconds / 1e6;
      metricsState.lastCompletedGpuFrameMs = gpuFrameMs;
    }

    gl.deleteQuery(pending.query);
    metricsState.pendingGpuQueries.splice(i, 1);

    emitMetrics(metricsState, {
      type: "gpu",
      frameId: pending.frameId,
      fps: computeAverageFps(metricsState.recentFrameDurationsMs),
      cpuFrameMs: metricsState.lastCpuFrameMs,
      cpuDrawMs: metricsState.lastCpuDrawMs,
      gpuFrameMs,
      gpuSupported: true,
      gpuDisjoint: Boolean(disjoint),
      timestampMs: nowMs(),
    });
  }
}

/**
 * Records CPU frame timing and emits a frame metrics event.
 *
 * @param {ImageTransformMetricsState} metricsState
 * @param {number} frameId
 * @param {number} frameStartMs
 * @param {number} drawStartMs
 * @param {number} drawEndMs
 * @param {number} frameEndMs
 */
function recordCpuMetrics(
  metricsState,
  frameId,
  frameStartMs,
  drawStartMs,
  drawEndMs,
  frameEndMs,
) {
  if (!metricsState.enabled) {
    return;
  }

  const cpuDrawMs = drawEndMs - drawStartMs;
  const cpuFrameMs = frameEndMs - frameStartMs;

  metricsState.lastCpuDrawMs = cpuDrawMs;
  metricsState.lastCpuFrameMs = cpuFrameMs;

  if (metricsState.lastFrameTimestampMs > 0) {
    const interFrameMs = frameEndMs - metricsState.lastFrameTimestampMs;
    if (interFrameMs > 0) {
      metricsState.recentFrameDurationsMs.push(interFrameMs);

      if (
        metricsState.recentFrameDurationsMs.length >
        metricsState.fpsSmoothingWindow
      ) {
        metricsState.recentFrameDurationsMs.shift();
      }
    }
  }

  metricsState.lastFrameTimestampMs = frameEndMs;

  emitMetrics(metricsState, {
    type: "frame",
    frameId,
    fps: computeAverageFps(metricsState.recentFrameDurationsMs),
    cpuFrameMs,
    cpuDrawMs,
    gpuFrameMs: metricsState.lastCompletedGpuFrameMs,
    gpuSupported: Boolean(metricsState.timerExtension),
    gpuDisjoint: false,
    timestampMs: frameEndMs,
  });
}

/**
 * Creates a renderer state object.
 *
 * Why a state object is returned:
 * it keeps all WebGL resources together so later calls can remain explicit
 * and argument-driven without reaching into outer scope.
 *
 * @param {HTMLCanvasElement} targetCanvas
 * @param {string} vertexShaderSource
 * @param {string} fragmentShaderSource
 * @param {{
 *   metrics?: {
 *     enabled?: boolean,
 *     callback?: (metrics: ImageTransformMetrics) => void,
 *     fpsSmoothingWindow?: number
 *   }
 * } | undefined} options
 * @returns {{
 *   canvas: HTMLCanvasElement,
 *   gl: WebGL2RenderingContext,
 *   program: WebGLProgram,
 *   vao: WebGLVertexArrayObject,
 *   vbo: WebGLBuffer,
 *   vertexCount: number,
 *   uniforms: Record<string, WebGLUniformLocation | null>,
 *   texture: WebGLTexture,
 *   toneCurveBuffer: Float32Array,
 *   colorCurveBuffer: Float32Array,
 *   metrics: ImageTransformMetricsState
 * }}
 */
function createImageTransformRenderer(
  targetCanvas,
  vertexShaderSource,
  fragmentShaderSource,
  options,
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

  const program = createGLSLProgram(
    gl,
    vertexShaderSource,
    fragmentShaderSource,
  );
  const quad = createFullscreenQuad(gl, program);
  const uniforms = getImageTransformUniformLocations(gl, program);
  const texture = createSourceTexture(gl);
  const metrics = createRendererMetricsState(gl, options?.metrics);

  gl.useProgram(program);

  if (uniforms.image) {
    gl.uniform1i(uniforms.image, 0);
  }

  return {
    canvas: targetCanvas,
    gl,
    program,
    vao: quad.vao,
    vbo: quad.vbo,
    vertexCount: quad.vertexCount,
    uniforms,
    texture,
    toneCurveBuffer: new Float32Array(TONE_CURVE_STEPS),
    colorCurveBuffer: new Float32Array(COLOR_CURVE_STEPS * 3),
    metrics,
  };
}

/**
 * Reads width and height from a supported image source.
 *
 * Why this helper exists:
 * videos, images, canvases, and ImageBitmaps expose their dimensions with
 * different property names. Centralizing that logic keeps the render path clean.
 *
 * @param {TexImageSource} source
 * @returns {{ width: number, height: number }}
 */
function getSourceDimensions(source) {
  if ("videoWidth" in source && "videoHeight" in source) {
    return {
      width: Math.max(1, source.videoWidth),
      height: Math.max(1, source.videoHeight),
    };
  }

  if ("naturalWidth" in source && "naturalHeight" in source) {
    return {
      width: Math.max(1, source.naturalWidth),
      height: Math.max(1, source.naturalHeight),
    };
  }

  if ("width" in source && "height" in source) {
    return {
      width: Math.max(1, source.width),
      height: Math.max(1, source.height),
    };
  }

  throw new Error(
    "Unsupported source type: could not determine source dimensions",
  );
}

/**
 * Resizes the target canvas to exactly match the source dimensions.
 *
 * Why this is done:
 * the caller asked for output size to match the original source size.
 * Setting the canvas to the source frame size avoids scaling and keeps a
 * 1:1 relationship between source pixels and output pixels.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {number} width
 * @param {number} height
 */
function resizeCanvasToSourceSize(canvas, width, height) {
  if (canvas.width !== width) {
    canvas.width = width;
  }

  if (canvas.height !== height) {
    canvas.height = height;
  }
}

/**
 * Applies the WebGL viewport for the current canvas size.
 *
 * Why this is separate:
 * resizing the canvas and updating the viewport are related but distinct
 * concerns. Keeping viewport logic explicit makes render setup easier to read.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {HTMLCanvasElement} canvas
 */
function applyViewportToCanvas(gl, canvas) {
  gl.viewport(0, 0, canvas.width, canvas.height);
}

/**
 * Uploads the current source image or video frame into the renderer texture.
 *
 * Why this is needed:
 * DOM media elements do not automatically update GPU memory. WebGL only sees
 * new pixel data after it is explicitly uploaded.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLTexture} texture
 * @param {TexImageSource} source
 */
function uploadSourceToTexture(gl, texture, source) {
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);

  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
}

/**
 * Converts a user-provided tone curve to the exact Float32Array shape required
 * by the shader.
 *
 * Why this helper exists:
 * shader uniforms require the correct flat typed-array layout. This keeps
 * validation and conversion out of the render path.
 *
 * @param {number[] | Float32Array | undefined} toneCurve
 * @param {Float32Array} targetBuffer
 * @returns {Float32Array}
 */
function writeToneCurveToBuffer(toneCurve, targetBuffer) {
  for (let i = 0; i < TONE_CURVE_STEPS; i++) {
    targetBuffer[i] = toneCurve?.[i] ?? 1.0;
  }

  return targetBuffer;
}

/**
 * Converts a user-provided color curve to the flat RGB array expected by
 * gl.uniform3fv.
 *
 * Supported input forms:
 * - flat array with 24 numbers
 * - nested array with 8 entries, each [r, g, b]
 *
 * Why this helper exists:
 * it lets the public API stay ergonomic while still feeding WebGL the exact
 * packed memory layout it needs.
 *
 * @param {number[] | Float32Array | Array<[number, number, number]> | undefined} colorCurve
 * @param {Float32Array} targetBuffer
 * @returns {Float32Array}
 */
function writeColorCurveToBuffer(colorCurve, targetBuffer) {
  if (!colorCurve) {
    for (let i = 0; i < targetBuffer.length; i++) {
      targetBuffer[i] = 1.0;
    }
    return targetBuffer;
  }

  if (Array.isArray(colorCurve) && Array.isArray(colorCurve[0])) {
    for (let i = 0; i < COLOR_CURVE_STEPS; i++) {
      const rgb = colorCurve[i] || [1, 1, 1];
      targetBuffer[i * 3 + 0] = rgb[0] ?? 1.0;
      targetBuffer[i * 3 + 1] = rgb[1] ?? 1.0;
      targetBuffer[i * 3 + 2] = rgb[2] ?? 1.0;
    }
    return targetBuffer;
  }

  for (let i = 0; i < targetBuffer.length; i++) {
    targetBuffer[i] = colorCurve[i] ?? 1.0;
  }

  return targetBuffer;
}

/**
 * Builds a normalized parameter object with safe defaults.
 *
 * Why this helper exists:
 * the caller should only provide the values they care about, while the
 * renderer always receives a complete and predictable parameter set.
 *
 * @param {object | undefined} params
 * @param {Float32Array} toneCurveBuffer
 * @param {Float32Array} colorCurveBuffer
 * @returns {{
 *   sharp: number,
 *   exposure: number,
 *   contrast: number,
 *   saturation: number,
 *   blur: number,
 *   toneCurve: Float32Array,
 *   colorCurve: Float32Array
 * }}
 */
function normalizeImageTransformParams(
  params,
  toneCurveBuffer,
  colorCurveBuffer,
) {
  const safeParams = params || {};

  return {
    sharp: safeParams.sharp ?? 0.0,
    exposure: safeParams.exposure ?? 0.0,
    contrast: safeParams.contrast ?? 1.0,
    saturation: safeParams.saturation ?? 1.0,
    blur: safeParams.blur ?? 0.0,
    toneCurve: writeToneCurveToBuffer(safeParams.toneCurve, toneCurveBuffer),
    colorCurve: writeColorCurveToBuffer(
      safeParams.colorCurve,
      colorCurveBuffer,
    ),
  };
}

/**
 * Uploads all per-frame uniforms.
 *
 * Why this is separate:
 * keeping uniform updates grouped in one place makes it easy to verify which
 * shader inputs depend on the current source frame and current user settings.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {Record<string, WebGLUniformLocation | null>} uniforms
 * @param {number} sourceWidth
 * @param {number} sourceHeight
 * @param {number} canvasWidth
 * @param {number} canvasHeight
 * @param {{
 *   sharp: number,
 *   exposure: number,
 *   contrast: number,
 *   saturation: number,
 *   blur: number,
 *   toneCurve: Float32Array,
 *   colorCurve: Float32Array
 * }} params
 */
function applyImageTransformUniforms(
  gl,
  uniforms,
  sourceWidth,
  sourceHeight,
  canvasWidth,
  canvasHeight,
  params,
) {
  if (uniforms.texelSize) {
    gl.uniform2f(uniforms.texelSize, 1 / sourceWidth, 1 / sourceHeight);
  }

  if (uniforms.canvasSize) {
    gl.uniform2f(uniforms.canvasSize, canvasWidth, canvasHeight);
  }

  if (uniforms.imageSize) {
    gl.uniform2f(uniforms.imageSize, sourceWidth, sourceHeight);
  }

  if (uniforms.sharp) {
    gl.uniform1f(uniforms.sharp, params.sharp);
  }

  if (uniforms.exposure) {
    gl.uniform1f(uniforms.exposure, params.exposure);
  }

  if (uniforms.contrast) {
    gl.uniform1f(uniforms.contrast, params.contrast);
  }

  if (uniforms.saturation) {
    gl.uniform1f(uniforms.saturation, params.saturation);
  }

  if (uniforms.blur) {
    gl.uniform1f(uniforms.blur, params.blur);
  }

  if (uniforms.toneCurve) {
    gl.uniform1fv(uniforms.toneCurve, params.toneCurve);
  }

  if (uniforms.colorCurve) {
    gl.uniform3fv(uniforms.colorCurve, params.colorCurve);
  }
}

/**
 * Checks whether a video source is currently usable for rendering.
 *
 * Why this helper exists:
 * video elements can exist before they actually have decoded frames.
 * Rendering too early causes invalid uploads or empty output.
 *
 * Non-video sources are treated as ready.
 *
 * @param {TexImageSource} source
 * @returns {boolean}
 */
function isRenderableSourceReady(source) {
  if (
    "readyState" in source &&
    "videoWidth" in source &&
    "videoHeight" in source
  ) {
    return (
      source.readyState >= 2 && source.videoWidth > 0 && source.videoHeight > 0
    );
  }

  const { width, height } = getSourceDimensions(source);
  return width > 0 && height > 0;
}

/**
 * Clears the canvas to opaque black.
 *
 * Why this is useful:
 * if rendering is skipped or the source is not ready yet, the target canvas
 * still ends up in a known state instead of showing stale pixels.
 *
 * @param {WebGL2RenderingContext} gl
 */
function clearRenderer(gl) {
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
}

/**
 * Returns a snapshot of the latest known renderer metrics.
 *
 * Why this helper exists:
 * some consumers prefer polling state instead of receiving callback events.
 *
 * @param {{
 *   metrics: ImageTransformMetricsState
 * }} renderer
 * @returns {{
 *   fps: number | null,
 *   cpuFrameMs: number | null,
 *   cpuDrawMs: number | null,
 *   gpuFrameMs: number | null,
 *   gpuSupported: boolean
 * }}
 */
function getImageTransformRendererMetrics(renderer) {
  return {
    fps: computeAverageFps(renderer.metrics.recentFrameDurationsMs),
    cpuFrameMs: renderer.metrics.lastCpuFrameMs,
    cpuDrawMs: renderer.metrics.lastCpuDrawMs,
    gpuFrameMs: renderer.metrics.lastCompletedGpuFrameMs,
    gpuSupported: Boolean(renderer.metrics.timerExtension),
  };
}

/**
 * Renders one processed frame from the source into the target canvas.
 *
 * This is the main entry point the library user should call.
 *
 * Why this function is designed as a one-frame render:
 * it keeps the library simple and predictable. The caller can use it for:
 * - a single image render
 * - manual redraws after parameter changes
 * - a requestAnimationFrame loop for video
 *
 * Metrics behavior:
 * - CPU metrics are emitted immediately for the current frame when enabled.
 * - GPU metrics are emitted later when the timer query resolves.
 *
 * @param {{
 *   canvas: HTMLCanvasElement,
 *   gl: WebGL2RenderingContext,
 *   program: WebGLProgram,
 *   vao: WebGLVertexArrayObject,
 *   vbo: WebGLBuffer,
 *   vertexCount: number,
 *   uniforms: Record<string, WebGLUniformLocation | null>,
 *   texture: WebGLTexture,
 *   toneCurveBuffer: Float32Array,
 *   colorCurveBuffer: Float32Array,
 *   metrics: ImageTransformMetricsState
 * }} renderer
 * @param {TexImageSource} source
 * @param {object} params
 * @returns {{ width: number, height: number }}
 */
function renderImageTransformFrame(renderer, source, params) {
  pollCompletedGpuMetrics(renderer.gl, renderer.metrics);

  if (!isRenderableSourceReady(source)) {
    clearRenderer(renderer.gl);
    return { width: 0, height: 0 };
  }

  const frameId = ++renderer.metrics.frameCounter;
  const frameStartMs = nowMs();

  const { width: sourceWidth, height: sourceHeight } =
    getSourceDimensions(source);
  const normalizedParams = normalizeImageTransformParams(
    params,
    renderer.toneCurveBuffer,
    renderer.colorCurveBuffer,
  );

  resizeCanvasToSourceSize(renderer.canvas, sourceWidth, sourceHeight);
  applyViewportToCanvas(renderer.gl, renderer.canvas);

  clearRenderer(renderer.gl);
  uploadSourceToTexture(renderer.gl, renderer.texture, source);

  renderer.gl.useProgram(renderer.program);
  renderer.gl.bindVertexArray(renderer.vao);

  applyImageTransformUniforms(
    renderer.gl,
    renderer.uniforms,
    sourceWidth,
    sourceHeight,
    renderer.canvas.width,
    renderer.canvas.height,
    normalizedParams,
  );

  const drawStartMs = nowMs();
  const gpuQueryState = beginGpuTimerQuery(renderer.gl, renderer.metrics);

  renderer.gl.drawArrays(renderer.gl.TRIANGLES, 0, renderer.vertexCount);

  endGpuTimerQuery(
    renderer.gl,
    renderer.metrics,
    gpuQueryState.query,
    gpuQueryState.didBegin,
    frameId,
  );

  renderer.gl.flush();

  const drawEndMs = nowMs();
  const frameEndMs = drawEndMs;

  recordCpuMetrics(
    renderer.metrics,
    frameId,
    frameStartMs,
    drawStartMs,
    drawEndMs,
    frameEndMs,
  );

  return {
    width: sourceWidth,
    height: sourceHeight,
  };
}

/**
 * Destroys all GPU resources held by the renderer.
 *
 * Why cleanup matters:
 * WebGL resources are not regular JavaScript memory. Releasing them when the
 * renderer is no longer needed prevents unnecessary GPU memory usage.
 *
 * @param {{
 *   gl: WebGL2RenderingContext,
 *   program: WebGLProgram,
 *   vao: WebGLVertexArrayObject,
 *   vbo: WebGLBuffer,
 *   texture: WebGLTexture,
 *   metrics: ImageTransformMetricsState
 * }} renderer
 */
function destroyImageTransformRenderer(renderer) {
  for (let i = 0; i < renderer.metrics.pendingGpuQueries.length; i++) {
    renderer.gl.deleteQuery(renderer.metrics.pendingGpuQueries[i].query);
  }

  renderer.metrics.pendingGpuQueries.length = 0;

  renderer.gl.deleteTexture(renderer.texture);
  renderer.gl.deleteBuffer(renderer.vbo);
  renderer.gl.deleteVertexArray(renderer.vao);
  renderer.gl.deleteProgram(renderer.program);
}

/**
 * @typedef {{
 *   type: "frame" | "gpu",
 *   frameId: number,
 *   fps: number | null,
 *   cpuFrameMs: number | null,
 *   cpuDrawMs: number | null,
 *   gpuFrameMs: number | null,
 *   gpuSupported: boolean,
 *   gpuDisjoint: boolean,
 *   timestampMs: number
 * }} ImageTransformMetrics
 */

/**
 * @typedef {{
 *   enabled: boolean,
 *   callback: ((metrics: ImageTransformMetrics) => void) | null,
 *   fpsSmoothingWindow: number,
 *   timerExtension: any,
 *   lastFrameTimestampMs: number,
 *   recentFrameDurationsMs: number[],
 *   pendingGpuQueries: Array<{ query: WebGLQuery, frameId: number }>,
 *   frameCounter: number,
 *   lastCompletedGpuFrameMs: number | null,
 *   lastCpuFrameMs: number | null,
 *   lastCpuDrawMs: number | null
 * }} ImageTransformMetricsState
 */

export {
  createImageTransformRenderer,
  destroyImageTransformRenderer,
  renderImageTransformFrame,
  clipToUV,
  imageAdjPipeline,
};
