



package com.example.gps

import android.widget.Toast
import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Looper
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.rememberMultiplePermissionsState
import com.google.android.gms.location.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import android.widget.ImageView
import android.util.Log

@OptIn(ExperimentalPermissionsApi::class)
class MainActivity : ComponentActivity() {
    private lateinit var fusedLocationClient: FusedLocationProviderClient

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        val imageView: ImageView = findViewById(R.id.contact_entry_image)
        Log.d("MainActivity", "ImageView is initialized: ${imageView != null}")
        imageView.setImageResource(R.drawable.image)





        setContent {
            GPSTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Column(
                        modifier = Modifier.fillMaxSize(),
                        verticalArrangement = Arrangement.Center,
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        ShowAppDescription()
                        RequestLocationPermission()
                    }
                }
            }
        }
    }

    @Composable
    fun ShowAppDescription() {
        Text(
            text = "Fake GPS detection system",
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.onBackground
        )
    }

    @Composable
    fun RequestLocationPermission() {
        val locationPermissionsState = rememberMultiplePermissionsState(
            listOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
            )
        )

        // 使用LaunchedEffect确保只在权限状态变化时执行操作
        LaunchedEffect(locationPermissionsState.allPermissionsGranted) {
            if (locationPermissionsState.allPermissionsGranted) {
                // 考虑使用协程来异步启动位置更新
                startLocationUpdates()
            } else {
                locationPermissionsState.launchMultiplePermissionRequest()
            }
        }

        // 使用简单的Text组件来提高性能，并确保只在权限状态变化时更新UI
        Text(if (locationPermissionsState.allPermissionsGranted) {
            "NOTE: This app will request location information permission"
        } else {
            "Location permission is required for this app to function."
        })
    }

    @Composable
    fun Greeting(message: String) {
        Text(text = message)
    }

    private fun startLocationUpdates() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            try {
                val locationRequest = LocationRequest.create().apply {
                    interval = 10000
                    fastestInterval = 5000
                    priority = Priority.PRIORITY_HIGH_ACCURACY
                }

                val locationCallback = object : LocationCallback() {
                    override fun onLocationResult(locationResult: LocationResult) {
                        for (location in locationResult.locations) {
                            // Use location data
                            val deviceId = fetchDeviceId()
                            sendLocationToServer(location.latitude, location.longitude, deviceId)
                        }
                    }
                }

                fusedLocationClient.requestLocationUpdates(
                    locationRequest,
                    locationCallback,
                    Looper.getMainLooper()
                )
            } catch (e: SecurityException) {
                // Handle exception if permission is not granted
            }
        }
    }

    private fun fetchDeviceId(): String {
        return Settings.Secure.getString(contentResolver, Settings.Secure.ANDROID_ID)
    }

    private fun sendLocationToServer(latitude: Double, longitude: Double, deviceId: String) {
        val client = OkHttpClient()

        val mediaType = "application/json; charset=utf-8".toMediaType()
        val json = """
        {
            "latitude": "$latitude",
            "longitude": "$longitude",
            "device_id": "$deviceId"
        }
        """.trimIndent()
        val requestBody = json.toRequestBody(mediaType)

        val request = Request.Builder()
            .url("https://809e-168-131-148-45.ngrok-free.app/submit_location")
            .post(requestBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                // Handle request failure
                runOnUiThread {
                    Toast.makeText(applicationContext, "Connection failed", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                // Handle successful response
                runOnUiThread {
                    if (response.isSuccessful) {
                        Toast.makeText(applicationContext, "Connection succeeded", Toast.LENGTH_SHORT).show()
                    } else {
                        Toast.makeText(applicationContext, "Connection failed: Server error", Toast.LENGTH_SHORT).show()
                    }
                }
                response.close()
            }
        })
    }
}

@Composable
fun GPSTheme(content: @Composable () -> Unit) {
    MaterialTheme {
        content()
    }
}
