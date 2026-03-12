(function() {
    console.log("IG Sobel: Toggle Button Version");

    if (window.sobelFilterEnabled === undefined) window.sobelFilterEnabled = false;
    const RENDER_SCALE = 0.5;

    const SobelEngine = {
        gl: null,
        program: null,
        texture: null,
        buffer: null,
        locations: {},
        sharedCanvas: document.createElement("canvas"),

        init() {
            if (this.gl) return true;
            const gl = this.sharedCanvas.getContext("webgl", { antialias:false, depth:false, alpha:false });
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
            const s = (t, src) => { const sh = gl.createShader(t); gl.shaderSource(sh, src); gl.compileShader(sh); return sh; };
            const p = gl.createProgram();
            gl.attachShader(p, s(gl.VERTEX_SHADER, vs));
            gl.attachShader(p, s(gl.FRAGMENT_SHADER, fs));
            gl.linkProgram(p);
            return p;
        },

        renderFrame(video, targetCanvas) {
            const gl = this.gl;
            if (video.videoWidth === 0) return;
            const targetW = Math.max(1, Math.floor(video.videoWidth * RENDER_SCALE));
            const targetH = Math.max(1, Math.floor(video.videoHeight * RENDER_SCALE));

            if (this.sharedCanvas.width !== targetW) {
                this.sharedCanvas.width = targetW;
                this.sharedCanvas.height = targetH;
            }

            gl.viewport(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight);
            gl.useProgram(this.program);
            gl.uniform2f(this.locations.res, gl.drawingBufferWidth, gl.drawingBufferHeight);

            gl.bindTexture(gl.TEXTURE_2D, this.texture);
            gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,video);

            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
            const pos = gl.getAttribLocation(this.program, "p");
            gl.enableVertexAttribArray(pos);
            gl.vertexAttribPointer(pos,2,gl.FLOAT,false,0,0);
            gl.drawArrays(gl.TRIANGLE_STRIP,0,4);

            const ctx = targetCanvas.getContext("2d", { alpha:false, desynchronized:true });
            if (targetCanvas.width!==targetW) { targetCanvas.width=targetW; targetCanvas.height=targetH; }
            ctx.drawImage(this.sharedCanvas,0,0);
        }
    };

    function modifyVideo(video) {
        if (video.dataset.sobelAttached) return;
        video.dataset.sobelAttached = "true";
        video.crossOrigin = "anonymous";

        const canvas = document.createElement("canvas");
        canvas.style.cssText = "position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;display:none;";
        video.insertAdjacentElement("afterend", canvas);

        video._sobel = { rAF:null, active:false, canvas };

        function loop() {
            if (!document.contains(video)) return;
            if (window.sobelFilterEnabled) {
                canvas.style.display="block"; video.style.opacity="0";
                if (!video._sobel.active && SobelEngine.init()) video._sobel.active=true;
                if (video._sobel.active && !video.paused && !video.ended)
                    SobelEngine.renderFrame(video, canvas);
            } else {
                canvas.style.display="none"; video.style.opacity="1"; video._sobel.active=false;
            }
            video._sobel.rAF=requestAnimationFrame(loop);
        }

        if (!video._sobel.rAF) loop();
    }

    function toggleFilter(on) {
        window.sobelFilterEnabled = !!on;
        updateButton();
    }

    // Create floating button
    function createButton() {
        if (document.getElementById("sobel-toggle-btn")) return;
        const btn = document.createElement("button");
        btn.id="sobel-toggle-btn";
        btn.innerText = window.sobelFilterEnabled ? "Sobel: ON" : "Sobel: OFF";
        btn.style.cssText = `
            position: fixed;
            top: 15px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 999999;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 12px;
            background: rgba(255,0,68,0.8);
            color: white;
            border: none;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        `;
        btn.onclick = () => { toggleFilter(!window.sobelFilterEnabled); };
        document.body.appendChild(btn);
    }

    function updateButton() {
        const btn = document.getElementById("sobel-toggle-btn");
        if (btn) {
            btn.innerText = window.sobelFilterEnabled ? "Sobel: ON" : "Sobel: OFF";
            btn.style.background = window.sobelFilterEnabled ? "rgba(255,0,68,0.8)" : "rgba(0,0,0,0.7)";
        }
    }

    window.enableSobelFilter = () => toggleFilter(true);
    window.disableSobelFilter = () => toggleFilter(false);

    createButton();

    const domObserver = new MutationObserver(muts => {
        muts.forEach(m => {
            m.addedNodes.forEach(node => {
                if (!(node instanceof Element)) return;
                if (node.tagName==="VIDEO") modifyVideo(node);
                node.querySelectorAll("video").forEach(modifyVideo);
            });
        });
    });
    domObserver.observe(document.body,{ childList:true, subtree:true });

    document.querySelectorAll("video").forEach(modifyVideo);
})();