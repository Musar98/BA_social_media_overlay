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

@SuppressLint("SetJavaScriptEnabled")
@Composable
fun InstagramWebView() {

    AndroidView(
        modifier = Modifier.fillMaxSize(),
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

//TODO update, this tries to draw red border around vids
private fun injectTestJavaScript(webView: WebView) {

    val js = """
        (function() {
            console.log("POC Injection Active");

            const videos = document.querySelectorAll("video");
            videos.forEach((video) => {
                video.style.border = "5px solid red";
            });

            return "Injected " + videos.length + " videos";
        })();
    """.trimIndent()

    webView.evaluateJavascript(js) { result ->
        println("JS Result: $result")
    }
}