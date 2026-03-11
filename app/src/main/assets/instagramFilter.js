(function () {

console.log("IG Sobel Filter Injection Started");

const MAX_HEIGHT = 720;

function modifyVideo(video) {

    if (video.dataset.sobelAttached) {
        return;
    }

    video.crossOrigin = "anonymous";
    video.dataset.sobelAttached = "true";

    const parent = video.parentElement;
    const canvas = document.createElement("canvas");

    parent.insertBefore(canvas, video);

    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    video.addEventListener("loadedmetadata", () => {

        let width = video.videoWidth;
        let height = video.videoHeight;

        if (height > MAX_HEIGHT) {
            const aspect = width / height;
            height = MAX_HEIGHT;
            width = Math.round(height * aspect);
        }

        canvas.width = width;
        canvas.height = height;

        canvas.style.width = video.clientWidth + "px";
        canvas.style.height = video.clientHeight + "px";

        video.style.display = "none";

        processFrame();
    });

    function processFrame() {

        if (video.paused || video.ended) {
            requestAnimationFrame(processFrame);
            return;
        }

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);

        const filtered = gradientImg(frame);

        ctx.putImageData(filtered, 0, 0);

        requestAnimationFrame(processFrame);
    }

    video.addEventListener("play", processFrame);
}

function processNode(node) {

    if (!(node instanceof Element)) return;

    if (node.tagName === "VIDEO") {
        modifyVideo(node);
    }

    node.querySelectorAll("video").forEach(modifyVideo);
}

document.querySelectorAll("video").forEach(modifyVideo);

const observer = new MutationObserver((mutations) => {

    mutations.forEach((mutation) => {
        mutation.addedNodes.forEach(processNode);
    });

});

observer.observe(document.body, {
    childList: true,
    subtree: true
});

////////////////////////////////////////////////////////////
// SOBEL FILTER (IDENTICAL TO REFERENCE IMPLEMENTATION)
////////////////////////////////////////////////////////////

// Buffer cache to avoid allocations
let bufferCache = null;

const kernel1 = new Float32Array([-1, 0, 1]);
const kernel2 = new Float32Array([1, 2, 1]);

function gradientImg(imageData) {

    const width = imageData.width;
    const height = imageData.height;
    const size = width * height;

    if (!bufferCache || bufferCache.size !== size) {

        bufferCache = {
            grayscale: new Float32Array(size),
            dx: new Float32Array(size),
            dxx: new Float32Array(size),
            dy: new Float32Array(size),
            dyy: new Float32Array(size),
            mag: new Float32Array(size),
            size
        };

    }

    toGrayscale(imageData, bufferCache.grayscale);

    applyHorizontalFilter(
        bufferCache.grayscale,
        kernel1,
        width,
        height,
        bufferCache.dx
    );

    applyVerticalFilter(
        bufferCache.dx,
        kernel2,
        width,
        height,
        bufferCache.dxx
    );

    applyVerticalFilter(
        bufferCache.grayscale,
        kernel1,
        width,
        height,
        bufferCache.dy
    );

    applyHorizontalFilter(
        bufferCache.dy,
        kernel2,
        width,
        height,
        bufferCache.dyy
    );

    magnitude(
        bufferCache.dxx,
        bufferCache.dyy,
        10,
        bufferCache.mag
    );

    return fromGrayscale(bufferCache.mag, imageData);
}

function toGrayscale(imageData, result) {

    const data = imageData.data;

    let dataIdx = 0;

    for (let i = 0; i < result.length; i++) {

        result[i] =
            (data[dataIdx++] +
             data[dataIdx++] +
             data[dataIdx++]) / 3;

        dataIdx++;

    }

}

function fromGrayscale(grayscale, imageData) {

    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;

    const output = new ImageData(width, height);
    const outputData = output.data;

    let dataIdx = 0;

    for (let i = 0; i < grayscale.length; i++) {

        const val = Math.min(Math.max(grayscale[i], 0), 255);

        outputData[dataIdx++] = val;
        outputData[dataIdx++] = val;
        outputData[dataIdx++] = val;

        // Copy original alpha channel
        outputData[dataIdx] = data[dataIdx];
        dataIdx++;
    }

    return output;
}

function applyHorizontalFilter(data, kernel, width, height, result) {

    const kernelSize = kernel.length;
    const kernelCenter = Math.floor(kernelSize / 2);

    result.fill(0);

    for (let y = 0; y < height; y++) {

        for (let x = kernelCenter; x < width - kernelCenter; x++) {

            let val = 0;

            for (let kx = 0; kx < kernelSize; kx++) {

                const srcX = x + kx - kernelCenter;
                const srcIndex = y * width + srcX;

                val += data[srcIndex] * kernel[kx];
            }

            result[y * width + x] = val;
        }
    }
}

function applyVerticalFilter(data, kernel, width, height, result) {

    const kernelSize = kernel.length;
    const kernelCenter = Math.floor(kernelSize / 2);

    result.fill(0);

    for (let y = kernelCenter; y < height - kernelCenter; y++) {

        for (let x = 0; x < width; x++) {

            let val = 0;

            for (let ky = 0; ky < kernelSize; ky++) {

                const srcY = y + ky - kernelCenter;
                const srcIndex = srcY * width + x;

                val += data[srcIndex] * kernel[ky];
            }

            result[y * width + x] = val;
        }
    }
}

function magnitude(x, y, scale, result) {

    for (let i = 0; i < x.length; i++) {

        result[i] =
            scale * Math.sqrt(x[i] * x[i] + y[i] * y[i]);

    }

}

})();