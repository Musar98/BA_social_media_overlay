// ImageAdjustmentsPipeline.ts
export const COLOR_ADJUSTMENT_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_image;
uniform float u_exposure;
uniform float u_contrast;
uniform float u_saturation;
uniform float u_imageMean;
uniform float u_toneCurve[8];
uniform vec3 u_colorCurve[8];

// Converts RGB to HSV.
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10f;

    return vec3(
        abs(q.z + (q.w - q.y) / (6.0f * d + e)),
        d / (q.x + e),
        q.x
    );
}

// Converts HSV back to RGB.
vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(
        fract(c.xxx + vec3(0.0f, 2.0f / 3.0f, 1.0f / 3.0f)) * 6.0f - 3.0f
    );

    return c.z * mix(
        vec3(1.0f),
        clamp(p - 1.0f, 0.0f, 1.0f),
        c.y
    );
}

// Samples the input image and returns RGB only.
vec3 sampleImage(vec2 uv) {
    return texture(u_image, uv).rgb;
}

// Applies exposure in photographic stops.
vec3 applyExposure(vec3 c, float exposureValue) {
    return clamp(c * exp2(exposureValue), 0.0f, 1.0f);
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

// Applies contrast around the externally supplied global image mean.
vec3 applyContrast(vec3 c, float contrastValue, float imgMean) {
    return clamp(
        c * contrastValue + vec3(imgMean) * (1.0f - contrastValue),
        0.0f,
        1.0f
    );
}

// Color-only part of the original pipeline.
vec3 applyAdjustments(vec3 color, float imgMean) {
    color = applyExposure(color, u_exposure);
    color = applySaturation(color, u_saturation);
    color = applyToneCurve(color);
    color = applyColorCurve(color);
    color = applyContrast(color, u_contrast, imgMean);

    return color;
}

void main() {
    vec3 color = sampleImage(v_uv);
    color = applyAdjustments(color, u_imageMean);
    outColor = vec4(clamp(color, 0.0f, 1.0f), 1.0f);
}`;

export const SHARPEN_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_image;
uniform vec2 u_texelSize;
uniform float u_sharp;

// Samples the input image and returns RGB only.
vec3 sampleImage(vec2 uv) {
    return texture(u_image, uv).rgb;
}

// Applies edge-based sharpening.
vec3 applySharpen(vec3 c, vec2 uv, vec2 texel, float sharpValue) {
    if(sharpValue <= 0.001f) {
        return c;
    }

    if(uv.x <= 0.5f * texel.x ||
       uv.x >= 1.0f - 0.5f * texel.x ||
       uv.y <= 0.5f * texel.y ||
       uv.y >= 1.0f - 0.5f * texel.y) {
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

void main() {
    vec3 color = sampleImage(v_uv);
    color = applySharpen(color, v_uv, u_texelSize, u_sharp);
    outColor = vec4(clamp(color, 0.0f, 1.0f), 1.0f);
}`;

export const BLUR_HORIZONTAL_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_image;
uniform vec2 u_texelSize;
uniform float u_blur;

vec3 sampleImage(vec2 uv) {
    return texture(u_image, uv).rgb;
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

void main() {
    vec3 sum = vec3(0.0f);
    float wsum = 0.0f;

    for(int x = -3; x <= 3; x++) {
        float w = gaussian1D(float(x), u_blur);
        vec2 suv = reflectUV(v_uv + vec2(float(x), 0.0f) * u_texelSize);
        sum += sampleImage(suv) * w;
        wsum += w;
    }

    outColor = vec4(clamp(sum / max(wsum, 1e-6f), 0.0f, 1.0f), 1.0f);
}`;

export const BLUR_VERTICAL_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_image;
uniform vec2 u_texelSize;
uniform float u_blur;

vec3 sampleImage(vec2 uv) {
    return texture(u_image, uv).rgb;
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

void main() {
    vec3 sum = vec3(0.0f);
    float wsum = 0.0f;

    for(int y = -3; y <= 3; y++) {
        float w = gaussian1D(float(y), u_blur);
        vec2 suv = reflectUV(v_uv + vec2(0.0f, float(y)) * u_texelSize);
        sum += sampleImage(suv) * w;
        wsum += w;
    }

    outColor = vec4(clamp(sum / max(wsum, 1e-6f), 0.0f, 1.0f), 1.0f);
}`;

export const COPY_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_image;

void main() {
    outColor = vec4(clamp(texture(u_image, v_uv).rgb, 0.0f, 1.0f), 1.0f);
}`;