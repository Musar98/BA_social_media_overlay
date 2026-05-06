export const IMAGE_ADJUSTMENTS_PIPELINE = `#version 300 es
precision highp float;

// Interpolated UV coordinates from the vertex shader.
in vec2 v_uv;

// Final output color for this fragment.
out vec4 outColor;

// Source image texture.
uniform sampler2D u_image;

// Size of one texel in UV space.
uniform vec2 u_texelSize;

// Size of the output canvas in pixels.
uniform vec2 u_canvasSize;

// Size of the source image in pixels.
uniform vec2 u_imageSize;

// User-controlled adjustment parameters.
uniform float u_sharp;       // Edge-based sharpening strength
uniform float u_exposure;    // Exposure in stops
uniform float u_contrast;    // Contrast multiplier
uniform float u_saturation;  // Saturation multiplier
uniform float u_blur;        // Blur radius multiplier
uniform float u_imageMean;   // Global image mean passed from CPU

// Piecewise tone curve weights.
uniform float u_toneCurve[8];

// Piecewise per-channel color curve weights.
uniform vec3 u_colorCurve[8];

// Converts RGB to HSV.
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

// Applies exposure in photographic stops.
vec3 applyExposure(vec3 c, float exposureValue) {
    return clamp(c * exp2(exposureValue), 0.0f, 1.0f);
}

// Applies contrast around a global midpoint (the image mean).
vec3 applyContrast(vec3 c, float contrastValue, float imgMean) {
    return clamp(c * contrastValue + vec3(imgMean) * (1.0 - contrastValue), 0.0, 1.0);
}

// Adjusts saturation in HSV space.
vec3 applySaturation(vec3 c, float satValue) {
    vec3 hsv = rgb2hsv(c);
    hsv.y = clamp(hsv.y * satValue, 0.0f, 1.0f);
    return clamp(hsv2rgb(hsv), 0.0f, 1.0f);
}

// Applies a piecewise tone curve across 8 equal input ranges.
vec3 applyToneCurve(vec3 c) {
    const int STEPS = 8;
    vec3 total = vec3(0.0f);
    float stepsF = float(STEPS);
    for(int i = 0; i < STEPS; i++) {
        float fi = float(i);
        vec3 seg = clamp(c - fi / stepsF, 0.0f, 1.0f / stepsF);
        total += seg * u_toneCurve[i];
    }
    return clamp(total, 0.0f, 1.0f);
}

// Applies a piecewise per-channel color curve across 8 equal input ranges.
vec3 applyColorCurve(vec3 c) {
    const int STEPS = 8;
    vec3 total = vec3(0.0f);
    float stepsF = float(STEPS);
    for(int i = 0; i < STEPS; i++) {
        float fi = float(i);
        vec3 seg = clamp(c - fi / stepsF, 0.0f, 1.0f / stepsF);
        total += seg * u_colorCurve[i];
    }
    return clamp(total, 0.0f, 1.0f);
}

// Samples the source image and returns RGB only.
vec3 sampleImage(vec2 uv) {
    return texture(u_image, uv).rgb;
}

// Applies edge-based sharpening.
vec3 applySharpen(vec3 c, vec2 uv, vec2 texel, float sharpValue) {
    if (sharpValue <= 0.001f) return c;
    if (uv.x <= 0.5f * texel.x || uv.x >= 1.0f - 0.5f * texel.x ||
        uv.y <= 0.5f * texel.y || uv.y >= 1.0f - 0.5f * texel.y) {
        return c;
    }
    vec3 sum =
        sampleImage(uv + texel * vec2(-1.0f, -1.0f)) +
        sampleImage(uv + texel * vec2( 0.0f, -1.0f)) +
        sampleImage(uv + texel * vec2( 1.0f, -1.0f)) +
        sampleImage(uv + texel * vec2(-1.0f,  0.0f)) +
        5.0f * c +
        sampleImage(uv + texel * vec2( 1.0f,  0.0f)) +
        sampleImage(uv + texel * vec2(-1.0f,  1.0f)) +
        sampleImage(uv + texel * vec2( 0.0f,  1.0f)) +
        sampleImage(uv + texel * vec2( 1.0f,  1.0f));
    vec3 degenerate = clamp(sum * (1.0f / 13.0f), 0.0f, 1.0f);
    return clamp(mix(degenerate, c, sharpValue), 0.0f, 1.0f);
}

// Converts canvas UV coordinates into image UV coordinates.
bool mapCanvasUVToImageUV(vec2 canvasUV, out vec2 imageUV) {
    float canvasAspect = u_canvasSize.x / u_canvasSize.y;
    float imageAspect = u_imageSize.x / u_imageSize.y;
    vec2 scale = vec2(1.0f);
    if(imageAspect > canvasAspect) {
        scale.y = canvasAspect / imageAspect;
    } else {
        scale.x = imageAspect / canvasAspect;
    }
    vec2 uv = (canvasUV - 0.5f) / scale + 0.5f;
    if(uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) return false;
    imageUV = uv;
    return true;
}

// Basic adjustment stack applied once to the final color.
vec3 adjustColor(vec3 color) {
    color = applyExposure(color, u_exposure);
    color = applyContrast(color, u_contrast, u_imageMean);
    color = applySaturation(color, u_saturation);
    color = applyToneCurve(color);
    color = applyColorCurve(color);
    return color;
}

vec2 reflectUV(vec2 uv) {
    uv = mod(uv, 2.0f);
    uv = abs(uv);
    return 1.0f - abs(1.0f - uv);
}

float gaussian1D(float x, float sigma) {
    sigma = max(sigma, 1e-6f);
    return exp(-(x * x) / (2.0f * sigma * sigma));
}

// Main logic: blur/sample RAW image, sharpen, then adjust colors.
vec3 applyOptimizedEffects(vec2 uv, vec2 texel) {
    vec3 color;
    if(u_blur > 0.001f) {
        vec3 sum = vec3(0.0f);
        float wsum = 0.0f;
        for(int y = -3; y <= 3; y++) {
            float wy = gaussian1D(float(y), u_blur);
            for(int x = -3; x <= 3; x++) {
                float wx = gaussian1D(float(x), u_blur);
                float w = wx * wy;
                vec2 suv = reflectUV(uv + vec2(float(x), float(y)) * texel);
                sum += sampleImage(suv) * w;
                wsum += w;
            }
        }
        color = sum / max(wsum, 1e-6f);
    } else {
        color = sampleImage(uv);
    }
    color = applySharpen(color, uv, texel, u_sharp);
    return adjustColor(color);
}

void main() {
    vec2 imageUV;
    if(!mapCanvasUVToImageUV(v_uv, imageUV)) {
        outColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }
    vec3 color = applyOptimizedEffects(imageUV, u_texelSize);
    outColor = vec4(clamp(color, 0.0f, 1.0f), 1.0f);
}`;
