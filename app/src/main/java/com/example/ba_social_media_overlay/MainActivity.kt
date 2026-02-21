package com.example.ba_social_media_overlay

import android.annotation.SuppressLint
import android.os.Bundle
import android.view.MotionEvent
import android.view.ViewGroup
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import com.example.ba_social_media_overlay.ui.theme.BA_social_media_overlayTheme

class MainActivity : ComponentActivity() {

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            BA_social_media_overlayTheme {
                InstagramWebView()
            }
        }
    }
}

//TODO rethink, with this option:
// Android does NOT process the video frames.
// Android does NOT draw overlays.
// All manipulation happens inside the webpage.
// The WebView is just a container.
// We inject js/css in func below!

@SuppressLint("SetJavaScriptEnabled")
@Composable
fun InstagramWebView() {

    AndroidView(
        modifier = Modifier
            .fillMaxSize()
            .systemBarsPadding(),
        factory = { context ->
            WebView(context).apply {

                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )

                isFocusable = true
                isFocusableInTouchMode = true
                requestFocus()

                setOnTouchListener { v, event ->
                    if (event.action == MotionEvent.ACTION_DOWN ||
                        event.action == MotionEvent.ACTION_UP
                    ) {
                        v.performClick()
                    }
                    false
                }

                settings.javaScriptEnabled = true
                settings.domStorageEnabled = true
                settings.mediaPlaybackRequiresUserGesture = false
                settings.loadsImagesAutomatically = true
                settings.useWideViewPort = true
                settings.loadWithOverviewMode = true

                webChromeClient = WebChromeClient()

                webViewClient = object : WebViewClient() {

                    override fun onPageFinished(view: WebView?, url: String?) {
                        super.onPageFinished(view, url)

                        postDelayed({
                            injectTestJavaScript(this@apply)
                        }, 2000)
                    }
                }

                loadUrl("https://www.instagram.com")
            }
        }
    )
}

//TODO
// Canvas video processing is BLOCKED by CORS (by instagram itself) in Android WebView
// so something like the sobel filter (accessing pixels does not work)
// DOMException → Canvas has been tainted by cross-origin data
private fun injectTestJavaScript(webView: WebView) {

    val js = """
        (function() {

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
                    element.style.filter = "grayscale(100%) contrast(200%) brightness(110%)";
                } else {
                    element.style.filter = "none";
                }
            }

            function updateFilterForAll() {
                document.querySelectorAll("video, img").forEach(el => {
                    el.style.filter = window.filterEnabled ? "grayscale(100%) contrast(200%) brightness(110%)" : "none";
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
                btn.style.left = "50%"; // center horizontally
                btn.style.transform = "translateX(-50%)"; // perfect center
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
        })();
    """.trimIndent()

    webView.evaluateJavascript(js, null)
}