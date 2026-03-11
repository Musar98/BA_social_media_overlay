(function () {

console.log("IG GPU Sobel Filter Injection Started");

function modifyVideo(video) {

   if (video.dataset.sobelAttached === video.src) return;
   video.dataset.sobelAttached = video.src;

    const parent = video.parentElement;

    const canvas = document.createElement("canvas");
    parent.insertBefore(canvas, video);

    const gl = canvas.getContext("webgl", { premultipliedAlpha: false });

    if (!gl) {
        console.error("WebGL not supported");
        return;
    }

    video.crossOrigin = "anonymous";

    video.addEventListener("loadedmetadata", () => {

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        canvas.style.width = video.clientWidth + "px";
        canvas.style.height = video.clientHeight + "px";

        video.style.display = "none";

        initGL(gl, video, canvas);
    });
}

////////////////////////////////////////////////////////////
// WEBGL INITIALIZATION
////////////////////////////////////////////////////////////

function initGL(gl, video, canvas) {

    const vertexSrc = `
    attribute vec2 position;
    varying vec2 vTex;

    void main() {
        vTex = (position + 1.0) * 0.5;
        gl_Position = vec4(position,0.0,1.0);
    }
    `;

    const fragmentSrc = `
    precision mediump float;

    varying vec2 vTex;

    uniform sampler2D tex;
    uniform vec2 texel;

    void main() {

        float tl = texture2D(tex, vTex + texel * vec2(-1.0,-1.0)).r;
        float tc = texture2D(tex, vTex + texel * vec2( 0.0,-1.0)).r;
        float tr = texture2D(tex, vTex + texel * vec2( 1.0,-1.0)).r;

        float ml = texture2D(tex, vTex + texel * vec2(-1.0, 0.0)).r;
        float mr = texture2D(tex, vTex + texel * vec2( 1.0, 0.0)).r;

        float bl = texture2D(tex, vTex + texel * vec2(-1.0, 1.0)).r;
        float bc = texture2D(tex, vTex + texel * vec2( 0.0, 1.0)).r;
        float br = texture2D(tex, vTex + texel * vec2( 1.0, 1.0)).r;

        float gx =
            -1.0 * tl + 1.0 * tr +
            -2.0 * ml + 2.0 * mr +
            -1.0 * bl + 1.0 * br;

        float gy =
             1.0 * tl + 2.0 * tc + 1.0 * tr +
            -1.0 * bl -2.0 * bc -1.0 * br;

        float g = length(vec2(gx,gy));

        gl_FragColor = vec4(vec3(g),1.0);
    }
    `;

    const program = createProgram(gl, vertexSrc, fragmentSrc);

    const position = gl.getAttribLocation(program, "position");
    const texelLoc = gl.getUniformLocation(program, "texel");

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);

    gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([
            -1,-1,
             1,-1,
            -1, 1,
             1, 1
        ]),
        gl.STATIC_DRAW
    );

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);

    // Flip video texture vertically (fix upside-down issue)
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);

    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

    function render() {

        if (video.paused || video.ended) {
            requestAnimationFrame(render);
            return;
        }

        gl.bindTexture(gl.TEXTURE_2D, texture);

        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            video
        );

        gl.viewport(0,0,canvas.width,canvas.height);

        gl.useProgram(program);

        gl.uniform2f(
            texelLoc,
            1.0 / canvas.width,
            1.0 / canvas.height
        );

        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);

        gl.enableVertexAttribArray(position);
        gl.vertexAttribPointer(position,2,gl.FLOAT,false,0,0);

        gl.drawArrays(gl.TRIANGLE_STRIP,0,4);

        requestAnimationFrame(render);
    }

    render();
}

////////////////////////////////////////////////////////////
// SHADER HELPERS
////////////////////////////////////////////////////////////

function createShader(gl,type,src){

    const shader = gl.createShader(type);

    gl.shaderSource(shader,src);
    gl.compileShader(shader);

    if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS)){
        console.error(gl.getShaderInfoLog(shader));
    }

    return shader;
}

function createProgram(gl,vsrc,fsrc){

    const program = gl.createProgram();

    const v = createShader(gl,gl.VERTEX_SHADER,vsrc);
    const f = createShader(gl,gl.FRAGMENT_SHADER,fsrc);

    gl.attachShader(program,v);
    gl.attachShader(program,f);

    gl.linkProgram(program);

    if(!gl.getProgramParameter(program,gl.LINK_STATUS)){
        console.error(gl.getProgramInfoLog(program));
    }

    return program;
}

////////////////////////////////////////////////////////////
// VIDEO DETECTION (INSTAGRAM FEED + REELS)
////////////////////////////////////////////////////////////

function processNode(node){

    if(!(node instanceof Element)) return;

    if(node.tagName==="VIDEO") modifyVideo(node);

    node.querySelectorAll("video").forEach(modifyVideo);
}

document.querySelectorAll("video").forEach(modifyVideo);

const observer = new MutationObserver(mutations=>{
    mutations.forEach(m=>{
        m.addedNodes.forEach(processNode);
    });
});

observer.observe(document.body,{
    childList:true,
    subtree:true
});

})();