(function () {
    console.log("IG Shared-GPU Sobel: UI-Preserving Version");

    const SobelEngine = {
        gl: null,
        program: null,
        texture: null,
        buffer: null,
        locations: {},
        sharedCanvas: document.createElement("canvas"),

        init() {
            if (this.gl) return true;
            const gl = this.sharedCanvas.getContext("webgl", { antialias: false, depth: false });
            if (!gl) return false;

            const vs = `attribute vec2 p; varying vec2 v; void main(){ v=(p+1.0)*0.5; gl_Position=vec4(p,0,1); }`;
            const fs = `
                precision mediump float;
                varying vec2 v;
                uniform sampler2D t;
                uniform vec2 res;
                void main() {
                    vec2 step = 1.0 / res;
                    float gX = -texture2D(t,v+step*vec2(-1,-1)).r + texture2D(t,v+step*vec2(1,-1)).r
                               -2.0*texture2D(t,v+step*vec2(-1,0)).r + 2.0*texture2D(t,v+step*vec2(1,0)).r
                               -texture2D(t,v+step*vec2(-1,1)).r + texture2D(t,v+step*vec2(1,1)).r;
                    float gY = texture2D(t,v+step*vec2(-1,-1)).r + 2.0*texture2D(t,v+step*vec2(0,-1)).r + texture2D(t,v+step*vec2(1,-1)).r
                               -texture2D(t,v+step*vec2(-1,1)).r - 2.0*texture2D(t,v+step*vec2(0,1)).r - texture2D(t,v+step*vec2(1,1)).r;
                    gl_FragColor = vec4(vec3(length(vec2(gX, gY))), 1.0);
                }
            `;

            this.program = this.createProg(gl, vs, fs);
            this.locations.res = gl.getUniformLocation(this.program, "res");
            this.buffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);

            this.texture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, this.texture);
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

            this.gl = gl;
            return true;
        },

        createProg(gl, vs, fs) {
            const s = (t, src) => {
                const sh = gl.createShader(t);
                gl.shaderSource(sh, src);
                gl.compileShader(sh);
                return sh;
            };
            const p = gl.createProgram();
            gl.attachShader(p, s(gl.VERTEX_SHADER, vs));
            gl.attachShader(p, s(gl.FRAGMENT_SHADER, fs));
            gl.linkProgram(p);
            return p;
        },

        renderFrame(video, targetCanvas) {
            const gl = this.gl;
            if (video.videoWidth === 0) return;

            if (this.sharedCanvas.width !== video.videoWidth) {
                this.sharedCanvas.width = video.videoWidth;
                this.sharedCanvas.height = video.videoHeight;
            }

            gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
            gl.useProgram(this.program);
            gl.uniform2f(this.locations.res, gl.drawingBufferWidth, gl.drawingBufferHeight);

            gl.bindTexture(gl.TEXTURE_2D, this.texture);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);

            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
            const pos = gl.getAttribLocation(this.program, "p");
            gl.enableVertexAttribArray(pos);
            gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 0, 0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            const targetCtx = targetCanvas.getContext("2d");
            if (targetCanvas.width !== video.videoWidth) {
                targetCanvas.width = video.videoWidth;
                targetCanvas.height = video.videoHeight;
            }
            targetCtx.drawImage(this.sharedCanvas, 0, 0);
        }
    };

    function modifyVideo(video) {
        if (video.dataset.sobelAttached) return;
        video.dataset.sobelAttached = "true";
        video.crossOrigin = "anonymous";

        // Create canvas
        const canvas = document.createElement("canvas");

        // CSS to overlay EXACTLY on top of the video, but behind UI
        // We use position absolute and match the video's object-fit behavior
        canvas.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 0;
            object-fit: ${getComputedStyle(video).objectFit};
        `;

        video.style.opacity = "0";
        video.style.position = "relative";
        video.style.zIndex = "-1";

        video.insertAdjacentElement('afterend', canvas);

        function update() {
            if (!document.contains(video)) return;

            if (!video.paused && !video.ended && SobelEngine.init()) {
                SobelEngine.renderFrame(video, canvas);
            }
            requestAnimationFrame(update);
        }
        update();
    }

    const observer = new MutationObserver(() => {
        document.querySelectorAll("video").forEach(modifyVideo);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    document.querySelectorAll("video").forEach(modifyVideo);
})();