package com.example.ba_social_media_overlay

import android.app.Activity
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import androidx.annotation.RequiresApi
import androidx.core.content.edit

class MainActivity : Activity() {

    private val REQUEST_CODE_SCREEN_CAPTURE = 1001

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val prefs = getSharedPreferences("overlay_prefs", MODE_PRIVATE)
        val alreadyGranted = prefs.getBoolean("media_projection_granted", false)

        if (!alreadyGranted) {
            val mgr = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            startActivityForResult(mgr.createScreenCaptureIntent(), REQUEST_CODE_SCREEN_CAPTURE)
        } else {
            val serviceIntent = Intent(this, ScreenCaptureService::class.java)
            startForegroundService(serviceIntent)
            finish()
        }
    }

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == REQUEST_CODE_SCREEN_CAPTURE && resultCode == RESULT_OK && data != null) {
            val prefs = getSharedPreferences("overlay_prefs", MODE_PRIVATE)
            prefs.edit { putBoolean("media_projection_granted", true) }

            val serviceIntent = Intent(this, ScreenCaptureService::class.java)
            serviceIntent.putExtra("resultCode", resultCode)
            serviceIntent.putExtra("data", data)
            startForegroundService(serviceIntent)
        }

        finish() // close MainActivity
    }
}