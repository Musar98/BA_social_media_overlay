(function(){var e={params:void 0},t={filterEnabled:!1};function n(){if(document.getElementById(`filter-toggle-btn`))return;let e=document.createElement(`button`);e.id=`filter-toggle-btn`,e.style.cssText=`
        position: fixed;
        top: 15px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 999999;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
        background: rgba(0, 120, 255, 0.8);
        color: white;
        border: none;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    `,e.onclick=()=>{t.filterEnabled=!t.filterEnabled,r()},document.body.appendChild(e),r()}function r(){let e=document.getElementById(`filter-toggle-btn`);e&&(e.innerText=t.filterEnabled?`Filter: ON`:`Filter: OFF`,e.style.background=t.filterEnabled?`rgba(0, 120, 255, 0.8)`:`rgba(0,0,0,0.7)`)}function i(e){let t=document.createElement(`canvas`);return t.style.cssText=`
    position:absolute;
    top:0;
    left:0;
    width:100%;
    height:100%;
    pointer-events:none;
    display:none;
    z-index:0;
  `,e.insertAdjacentElement(`afterend`,t),t}function a(e,t,n){let r=e.createShader(t);if(!r)throw Error(`Failed to create shader object`);if(e.shaderSource(r,n),e.compileShader(r),!e.getShaderParameter(r,e.COMPILE_STATUS)){let t=e.getShaderInfoLog(r)||`Unknown shader compile nvm error`;throw e.deleteShader(r),Error(`Shader compilation failed.\n\n${t}\n\nSource:\n${n}`)}return r}function o(e,t,n){let r=a(e,e.VERTEX_SHADER,t),i=a(e,e.FRAGMENT_SHADER,n),o=e.createProgram();if(!o)throw e.deleteShader(r),e.deleteShader(i),Error(`Failed to create WebGL program`);e.attachShader(o,r),e.attachShader(o,i),e.linkProgram(o);let s=e.getProgramParameter(o,e.LINK_STATUS);if(e.deleteShader(r),e.deleteShader(i),!s){let t=e.getProgramInfoLog(o)||`Unknown program link error`;throw e.deleteProgram(o),Error(`Program linking failed.\n\n${t}`)}return o}function s(e,t){let n=new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),r=e.createVertexArray();if(!r)throw Error(`Failed to create vertex array object`);let i=e.createBuffer();if(!i)throw e.deleteVertexArray(r),Error(`Failed to create vertex buffer object`);e.bindVertexArray(r),e.bindBuffer(e.ARRAY_BUFFER,i),e.bufferData(e.ARRAY_BUFFER,n,e.STATIC_DRAW);let a=e.getAttribLocation(t,`a_pos`);if(a<0)throw e.deleteBuffer(i),e.deleteVertexArray(r),Error(`Shader attribute "a_pos" was not found`);return e.enableVertexAttribArray(a),e.vertexAttribPointer(a,2,e.FLOAT,!1,0,0),e.bindVertexArray(null),e.bindBuffer(e.ARRAY_BUFFER,null),{vao:r,vbo:i,vertexCount:6}}function c(e,t){return{image:e.getUniformLocation(t,`u_image`),texelSize:e.getUniformLocation(t,`u_texelSize`),canvasSize:e.getUniformLocation(t,`u_canvasSize`),imageSize:e.getUniformLocation(t,`u_imageSize`),sharp:e.getUniformLocation(t,`u_sharp`),exposure:e.getUniformLocation(t,`u_exposure`),contrast:e.getUniformLocation(t,`u_contrast`),saturation:e.getUniformLocation(t,`u_saturation`),blur:e.getUniformLocation(t,`u_blur`),toneCurve:e.getUniformLocation(t,`u_toneCurve`),colorCurve:e.getUniformLocation(t,`u_colorCurve`)}}function l(e){let t=e.createTexture();if(!t)throw Error(`Failed to create texture`);return e.activeTexture(e.TEXTURE0),e.bindTexture(e.TEXTURE_2D,t),e.texParameteri(e.TEXTURE_2D,e.TEXTURE_WRAP_S,e.CLAMP_TO_EDGE),e.texParameteri(e.TEXTURE_2D,e.TEXTURE_WRAP_T,e.CLAMP_TO_EDGE),e.texParameteri(e.TEXTURE_2D,e.TEXTURE_MIN_FILTER,e.LINEAR),e.texParameteri(e.TEXTURE_2D,e.TEXTURE_MAG_FILTER,e.LINEAR),e.bindTexture(e.TEXTURE_2D,null),t}function u(){return typeof performance<`u`&&typeof performance.now==`function`?performance.now():Date.now()}function d(e,t){let n=!!t?.enabled;return{enabled:n,callback:typeof t?.callback==`function`?t.callback:null,fpsSmoothingWindow:Math.max(1,t?.fpsSmoothingWindow??30),timerExtension:n?e.getExtension(`EXT_disjoint_timer_query_webgl2`):null,lastFrameTimestampMs:0,recentFrameDurationsMs:[],pendingGpuQueries:[],frameCounter:0,lastCompletedGpuFrameMs:null,lastCpuFrameMs:null,lastCpuDrawMs:null}}function f(e,t){if(!t.enabled||!t.timerExtension||t.pendingGpuQueries.length>=8)return{query:null,didBegin:!1};let n=e.createQuery();return n?(e.beginQuery(t.timerExtension.TIME_ELAPSED_EXT,n),{query:n,didBegin:!0}):{query:null,didBegin:!1}}function p(e,t,n,r,i){!r||!n||!t.timerExtension||(e.endQuery(t.timerExtension.TIME_ELAPSED_EXT),t.pendingGpuQueries.push({query:n,frameId:i}))}function m(e){if(e.length===0)return null;let t=0;for(let n=0;n<e.length;n++)t+=e[n];return t<=0?null:1e3/(t/e.length)}function h(e,t){!e.enabled||!e.callback||e.callback(t)}function g(e,t){if(!t.enabled||!t.timerExtension)return;let n=t.timerExtension,r=e.getParameter(n.GPU_DISJOINT_EXT);for(let n=0;n<t.pendingGpuQueries.length;){let i=t.pendingGpuQueries[n];if(!e.getQueryParameter(i.query,e.QUERY_RESULT_AVAILABLE)){n+=1;continue}let a=null;r||(a=e.getQueryParameter(i.query,e.QUERY_RESULT)/1e6,t.lastCompletedGpuFrameMs=a),e.deleteQuery(i.query),t.pendingGpuQueries.splice(n,1),h(t,{type:`gpu`,frameId:i.frameId,fps:m(t.recentFrameDurationsMs),cpuFrameMs:t.lastCpuFrameMs,cpuDrawMs:t.lastCpuDrawMs,gpuFrameMs:a,gpuSupported:!0,gpuDisjoint:!!r,timestampMs:u()})}}function _(e,t,n,r,i,a){if(!e.enabled)return;let o=i-r,s=a-n;if(e.lastCpuDrawMs=o,e.lastCpuFrameMs=s,e.lastFrameTimestampMs>0){let t=a-e.lastFrameTimestampMs;t>0&&(e.recentFrameDurationsMs.push(t),e.recentFrameDurationsMs.length>e.fpsSmoothingWindow&&e.recentFrameDurationsMs.shift())}e.lastFrameTimestampMs=a,h(e,{type:`frame`,frameId:t,fps:m(e.recentFrameDurationsMs),cpuFrameMs:s,cpuDrawMs:o,gpuFrameMs:e.lastCompletedGpuFrameMs,gpuSupported:!!e.timerExtension,gpuDisjoint:!1,timestampMs:a})}function v(e,t,n,r){let i=e.getContext(`webgl2`,{alpha:!1,depth:!1,stencil:!1,antialias:!1,preserveDrawingBuffer:!1,premultipliedAlpha:!0,powerPreference:`high-performance`});if(!i)throw Error(`WebGL2 is not available for the target canvas`);let a=o(i,t,n),u=s(i,a),f=c(i,a),p=l(i),m=d(i,r?.metrics);return i.useProgram(a),f.image&&i.uniform1i(f.image,0),{canvas:e,gl:i,program:a,vao:u.vao,vbo:u.vbo,vertexCount:u.vertexCount,uniforms:f,texture:p,toneCurveBuffer:new Float32Array(8),colorCurveBuffer:new Float32Array(24),metrics:m}}function y(e){if(`videoWidth`in e&&`videoHeight`in e)return{width:Math.max(1,e.videoWidth),height:Math.max(1,e.videoHeight)};if(`naturalWidth`in e&&`naturalHeight`in e)return{width:Math.max(1,e.naturalWidth),height:Math.max(1,e.naturalHeight)};if(`width`in e&&`height`in e)return{width:Math.max(1,e.width),height:Math.max(1,e.height)};throw Error(`Unsupported source type: could not determine source dimensions`)}function b(e,t,n){e.width!==t&&(e.width=t),e.height!==n&&(e.height=n)}function x(e,t){e.viewport(0,0,t.width,t.height)}function S(e,t,n){e.activeTexture(e.TEXTURE0),e.bindTexture(e.TEXTURE_2D,t),e.pixelStorei(e.UNPACK_FLIP_Y_WEBGL,!1),e.texImage2D(e.TEXTURE_2D,0,e.RGBA,e.RGBA,e.UNSIGNED_BYTE,n)}function C(e,t){for(let n=0;n<8;n++)t[n]=e?.[n]??1;return t}function w(e,t){if(!e){for(let e=0;e<t.length;e++)t[e]=1;return t}if(Array.isArray(e)&&Array.isArray(e[0])){for(let n=0;n<8;n++){let r=e[n]||[1,1,1];t[n*3+0]=r[0]??1,t[n*3+1]=r[1]??1,t[n*3+2]=r[2]??1}return t}for(let n=0;n<t.length;n++)t[n]=e[n]??1;return t}function T(e,t,n){let r=e||{};return{sharp:r.sharp??0,exposure:r.exposure??0,contrast:r.contrast??1,saturation:r.saturation??1,blur:r.blur??0,toneCurve:C(r.toneCurve,t),colorCurve:w(r.colorCurve,n)}}function E(e,t,n,r,i,a,o){t.texelSize&&e.uniform2f(t.texelSize,1/n,1/r),t.canvasSize&&e.uniform2f(t.canvasSize,i,a),t.imageSize&&e.uniform2f(t.imageSize,n,r),t.sharp&&e.uniform1f(t.sharp,o.sharp),t.exposure&&e.uniform1f(t.exposure,o.exposure),t.contrast&&e.uniform1f(t.contrast,o.contrast),t.saturation&&e.uniform1f(t.saturation,o.saturation),t.blur&&e.uniform1f(t.blur,o.blur),t.toneCurve&&e.uniform1fv(t.toneCurve,o.toneCurve),t.colorCurve&&e.uniform3fv(t.colorCurve,o.colorCurve)}function D(e){if(`readyState`in e&&`videoWidth`in e&&`videoHeight`in e)return e.readyState>=2&&e.videoWidth>0&&e.videoHeight>0;let{width:t,height:n}=y(e);return t>0&&n>0}function O(e){e.clearColor(0,0,0,1),e.clear(e.COLOR_BUFFER_BIT)}function k(e,t,n){if(g(e.gl,e.metrics),!D(t))return O(e.gl),{width:0,height:0};let r=++e.metrics.frameCounter,i=u(),{width:a,height:o}=y(t),s=T(n,e.toneCurveBuffer,e.colorCurveBuffer);b(e.canvas,a,o),x(e.gl,e.canvas),O(e.gl),S(e.gl,e.texture,t),e.gl.useProgram(e.program),e.gl.bindVertexArray(e.vao),E(e.gl,e.uniforms,a,o,e.canvas.width,e.canvas.height,s);let c=u(),l=f(e.gl,e.metrics);e.gl.drawArrays(e.gl.TRIANGLES,0,e.vertexCount),p(e.gl,e.metrics,l.query,l.didBegin,r),e.gl.flush();let d=u(),m=d;return _(e.metrics,r,i,c,d,m),{width:a,height:o}}function A(e){for(let t=0;t<e.metrics.pendingGpuQueries.length;t++)e.gl.deleteQuery(e.metrics.pendingGpuQueries[t].query);e.metrics.pendingGpuQueries.length=0,e.gl.deleteTexture(e.texture),e.gl.deleteBuffer(e.vbo),e.gl.deleteVertexArray(e.vao),e.gl.deleteProgram(e.program)}var j=`#version 300 es
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
}`,M=`#version 300 es
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
}`,N=null;function P(e){return N&&F(),N=v(e,j,M,{metrics:{enabled:!1}}),N}function F(){if(!N)return;let e=N?.gl;e&&e.getExtension(`WEBGL_lose_context`)?.loseContext(),A(N),N=null}var I=`/static_resources/webworker_v1/init_script/aiWorker.js`,L=null,R=null;function z(n,r,i=30){let a=P(r),o=0;try{R=new Worker(I),R.onmessage=t=>{let{aiParams:n,error:r}=t.data;r&&console.error(`AI Worker error:`,r),n&&(e.params=n)}}catch(e){console.error(`Failed to create AI worker:`,e)}function s(){if(!R||n.readyState<2)return;let e=new OffscreenCanvas(n.videoWidth,n.videoHeight);e.getContext(`2d`).drawImage(n,0,0);let t=e.transferToImageBitmap();R.postMessage({bitmap:t},[t])}function c(){if(!document.contains(n)){B();return}t.filterEnabled?(r.style.display=`block`,n.style.opacity=`0`,!n.paused&&!n.ended&&(o++,o%i===0&&s(),console.log(`[Render Loop] Rendering with params:`,[e.params?.sharp,e.params?.exposure,e.params?.contrast,e.params?.saturation,e.params?.blur]),k(a,n,e.params))):(r.style.display=`none`,n.style.opacity=`1`),L=requestAnimationFrame(c)}c()}function B(){L&&cancelAnimationFrame(L),L=null,R?.terminate(),R=null,F()}var V=null,H=null;function U(e){e!==V&&(document.querySelectorAll(`video`).forEach(t=>{t!==e&&(t.dataset.filterAttached=`true`)}),e.dataset.filterAttached=`true`,e.crossOrigin=`anonymous`,V=e,H=i(e),z(e,H))}function W(){V&&=(B(),null),H&&=(H.remove(),null)}var G=null,K=new IntersectionObserver(e=>{e.forEach(e=>{let t=e.target;e.isIntersecting&&e.intersectionRatio>.6&&G!==t&&(W(),G=t,U(t))})},{threshold:[.6]});function q(){new MutationObserver(e=>{e.forEach(e=>{e.addedNodes.forEach(e=>{e instanceof HTMLVideoElement?K.observe(e):e instanceof Element&&e.querySelectorAll(`video`).forEach(e=>{K.observe(e)})})})}).observe(document.body,{childList:!0,subtree:!0}),document.querySelectorAll(`video`).forEach(e=>{K.observe(e)})}async function J(){n(),q()}J().catch(console.error)})();