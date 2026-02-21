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
// clear up if this is sufficient or we want to evaluate more options
// for example:
// Native Android Video Processing (Much Stronger)
// Instead of:
// WebView → JavaScript → Canvas
// Try:
// Capture video using MediaCodec / ExoPlayer
// Process frames in Kotlin
// Apply Sobel / other filterin natively
// Render overlay
// that would bypass CORS (ig)
// but changes Architecture
// might be Technically more powerful (almost surely since allows full kt interactivity)

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
private fun injectTestJavaScript(
    webView: WebView,
    filterCss: String = "grayscale(100%) contrast(200%) brightness(110%)"
) {

    val context = webView.context
    val js = context.assets.open("instagramFilter.js").bufferedReader().use { it.readText() }
    val finalJs = "window.dynamicFilterCss = \"${filterCss.replace("\"", "\\\"")}\";\n$js"

    webView.evaluateJavascript(finalJs, null)
}