package com.example.ba_social_media_overlay


import android.view.WindowManager
import android.webkit.WebView
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge

object WebAppBootstrap {

    fun init(activity: ComponentActivity) {
        activity.enableEdgeToEdge()

        activity.window.setFlags(
            WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED,
            WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED
        )

        WebView.setWebContentsDebuggingEnabled(true)
    }
}