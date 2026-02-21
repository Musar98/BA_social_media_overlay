(function(filterCss) {

    console.log("CSS Filter Injection Active with Toggle");

    // state of toggle filter (on/off)
    if (window.filterEnabled === undefined) {
        window.filterEnabled = true;
    }

    // handling css filter
    function applyFilter(element) {
        if (element.dataset.filtered) return;
        element.dataset.filtered = "true";

        element.style.transition = "filter 0.3s ease";
        if (window.filterEnabled) {
            element.style.filter = filterCss;
        } else {
            element.style.filter = "none";
        }
    }

    function updateFilterForAll() {
        document.querySelectorAll("video, img").forEach(el => {
            el.style.filter = window.filterEnabled ? filterCss : "none";
        });
    }

    function processNode(node) {
        if (node.tagName === "VIDEO" || node.tagName === "IMG") {
            applyFilter(node);
        }

        if (node.querySelectorAll) {
            node.querySelectorAll("video, img").forEach(applyFilter);
        }
    }

    // init pass
    document.querySelectorAll("video, img").forEach(processNode);

    // observe dyn content
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach(processNode);
        });
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    // Button for toggling filter
    if (!document.getElementById("filterToggleButton")) {
        const btn = document.createElement("button");
        btn.id = "filterToggleButton";
        btn.innerText = "Toggle Filter";
        btn.style.position = "fixed";
        btn.style.top = "10px";
        btn.style.left = "50%";
        btn.style.transform = "translateX(-50%)";
        btn.style.zIndex = "9999";
        btn.style.padding = "8px 12px";
        btn.style.background = "#ff0044";
        btn.style.color = "#fff";
        btn.style.border = "none";
        btn.style.borderRadius = "5px";
        btn.style.cursor = "pointer";
        btn.style.fontSize = "14px";
        btn.style.boxShadow = "0 2px 5px rgba(0,0,0,0.3)";
        btn.onclick = function() {
            window.filterEnabled = !window.filterEnabled;
            updateFilterForAll();
            console.log("Filter Enabled:", window.filterEnabled);
        };

        document.body.appendChild(btn);
    }

    return "CSS Filter + Toggle Injected";

})(window.dynamicFilterCss || "grayscale(100%) contrast(200%) brightness(110%)");