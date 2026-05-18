// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";
import { ImageTransformRenderer } from "../../src/transformation/transformations";

function createMockGl() {
  const gl = {
    VERTEX_SHADER: 1,
    FRAGMENT_SHADER: 2,
    COMPILE_STATUS: 3,
    LINK_STATUS: 4,
    ARRAY_BUFFER: 5,
    STATIC_DRAW: 6,
    FLOAT: 7,
    TEXTURE0: 8,
    TEXTURE_2D: 9,
    TEXTURE_WRAP_S: 10,
    TEXTURE_WRAP_T: 11,
    CLAMP_TO_EDGE: 12,
    TEXTURE_MIN_FILTER: 13,
    TEXTURE_MAG_FILTER: 14,
    LINEAR: 15,
    UNPACK_FLIP_Y_WEBGL: 16,
    RGBA: 17,
    UNSIGNED_BYTE: 18,
    COLOR_BUFFER_BIT: 19,
    TRIANGLES: 20,
    createShader: vi.fn(() => ({})),
    shaderSource: vi.fn(),
    compileShader: vi.fn(),
    getShaderParameter: vi.fn(() => true),
    getShaderInfoLog: vi.fn(() => ""),
    deleteShader: vi.fn(),
    createProgram: vi.fn(() => ({})),
    attachShader: vi.fn(),
    linkProgram: vi.fn(),
    getProgramParameter: vi.fn(() => true),
    getProgramInfoLog: vi.fn(() => ""),
    deleteProgram: vi.fn(),
    createVertexArray: vi.fn(() => ({})),
    createBuffer: vi.fn(() => ({})),
    bindVertexArray: vi.fn(),
    bindBuffer: vi.fn(),
    bufferData: vi.fn(),
    getAttribLocation: vi.fn(() => 0),
    enableVertexAttribArray: vi.fn(),
    vertexAttribPointer: vi.fn(),
    getUniformLocation: vi.fn(() => ({})),
    createTexture: vi.fn(() => ({})),
    activeTexture: vi.fn(),
    bindTexture: vi.fn(),
    texParameteri: vi.fn(),
    useProgram: vi.fn(),
    uniform1i: vi.fn(),
    clearColor: vi.fn(),
    clear: vi.fn(),
    pixelStorei: vi.fn(),
    texImage2D: vi.fn(),
    texSubImage2D: vi.fn(),
    viewport: vi.fn(),
    uniform2f: vi.fn(),
    uniform1f: vi.fn(),
    uniform1fv: vi.fn(),
    uniform3fv: vi.fn(),
    drawArrays: vi.fn(),
    getExtension: vi.fn(() => null),
    deleteTexture: vi.fn(),
    deleteBuffer: vi.fn(),
    deleteVertexArray: vi.fn(),
  };

  return gl;
}

describe("ImageTransformRenderer", () => {
  it("does not clear before valid full-screen draws", () => {
    const gl = createMockGl();
    const canvas = document.createElement("canvas");
    canvas.getContext = vi.fn(() => gl as any);

    const renderer = new ImageTransformRenderer(canvas, "vertex", "fragment");
    const source = document.createElement("canvas");
    source.width = 10;
    source.height = 10;

    renderer.renderFrame(source, {});

    expect(gl.clear).not.toHaveBeenCalled();
    expect(gl.drawArrays).toHaveBeenCalled();
  });

  it("still clears when the video source is not renderable", () => {
    const gl = createMockGl();
    const canvas = document.createElement("canvas");
    canvas.getContext = vi.fn(() => gl as any);

    const renderer = new ImageTransformRenderer(canvas, "vertex", "fragment");
    const source = document.createElement("video");

    renderer.renderFrame(source, {});

    expect(gl.clear).toHaveBeenCalledWith(gl.COLOR_BUFFER_BIT);
    expect(gl.drawArrays).not.toHaveBeenCalled();
  });
});
