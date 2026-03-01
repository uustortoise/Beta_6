package com.facephysio;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.media.Image;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

/**
 * Main Activity for FacePhysio Vital Signs Monitor
 * 
 * Uses CameraX for camera capture and Chaquopy to run Python/pyVHR
 * for signal processing and vital sign calculation.
 */
public class MainActivity extends AppCompatActivity {
    
    private static final String TAG = "FacePhysio";
    private static final int CAMERA_PERMISSION_REQUEST = 100;
    private static final int ANALYSIS_INTERVAL_MS = 2000; // Analyze every 2 seconds
    
    // UI Components
    private PreviewView previewView;
    private TextView tvStatus;
    private TextView tvHeartRate;
    private TextView tvSpO2;
    private TextView tvRespiration;
    private TextView tvHRV;
    private TextView tvSignalQuality;
    private TextView tvDebugInfo;
    private Button btnStart;
    private Button btnStop;
    private ProgressBar progressBar;
    private View qualityIndicator;
    
    // Camera
    private ProcessCameraProvider cameraProvider;
    private final Executor cameraExecutor = Executors.newSingleThreadExecutor();
    private boolean isAnalyzing = false;
    
    // Python/Chaquopy
    private Python py;
    private PyObject vitalsModule;
    
    // Handler for UI updates
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private Runnable analysisRunnable;
    
    // Frame dimensions
    private int frameWidth = 640;
    private int frameHeight = 480;
    private static final int TARGET_WIDTH = 320;  // Downsample for performance
    private static final int TARGET_HEIGHT = 240;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        initializeViews();
        initializePython();
        setupButtons();
        checkPermissions();
    }
    
    private void initializeViews() {
        previewView = findViewById(R.id.previewView);
        tvStatus = findViewById(R.id.tvStatus);
        tvHeartRate = findViewById(R.id.tvHeartRate);
        tvSpO2 = findViewById(R.id.tvSpO2);
        tvRespiration = findViewById(R.id.tvRespiration);
        tvHRV = findViewById(R.id.tvHRV);
        tvSignalQuality = findViewById(R.id.tvSignalQuality);
        tvDebugInfo = findViewById(R.id.tvDebugInfo);
        btnStart = findViewById(R.id.btnStart);
        btnStop = findViewById(R.id.btnStop);
        progressBar = findViewById(R.id.progressBar);
        qualityIndicator = findViewById(R.id.qualityIndicator);
        
        updateUIState(false);
    }
    
    private void initializePython() {
        try {
            if (!Python.isStarted()) {
                Python.start(new AndroidPlatform(this));
            }
            py = Python.getInstance();
            vitalsModule = py.getModule("vitals_processor");
            
            // Initialize processor with 30fps and 5-second buffer
            PyObject result = vitalsModule.callAttr("initialize", 30, 5);
            Log.i(TAG, "Python/pyVHR initialized successfully");
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize Python", e);
            tvStatus.setText("Error: Python initialization failed");
            tvStatus.setTextColor(Color.RED);
        }
    }
    
    private void setupButtons() {
        btnStart.setOnClickListener(v -> startMeasurement());
        btnStop.setOnClickListener(v -> stopMeasurement());
        
        // Periodic analysis runnable
        analysisRunnable = new Runnable() {
            @Override
            public void run() {
                if (isAnalyzing) {
                    analyzeAndUpdate();
                    mainHandler.postDelayed(this, ANALYSIS_INTERVAL_MS);
                }
            }
        };
    }
    
    private void checkPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) 
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, 
                    new String[]{Manifest.permission.CAMERA}, 
                    CAMERA_PERMISSION_REQUEST);
        } else {
            startCamera();
        }
    }
    
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, 
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }
    
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = 
                ProcessCameraProvider.getInstance(this);
        
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                
                // Preview
                Preview preview = new Preview.Builder()
                        .setTargetResolution(new android.util.Size(frameWidth, frameHeight))
                        .build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());
                
                // Image analysis
                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setTargetResolution(new android.util.Size(frameWidth, frameHeight))
                        .build();
                
                imageAnalysis.setAnalyzer(cameraExecutor, this::processFrame);
                
                // Front camera
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                        .build();
                
                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
                
                tvStatus.setText("Camera ready - Press Start");
                tvStatus.setTextColor(Color.GREEN);
                
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Camera start failed", e);
                tvStatus.setText("Camera error");
                tvStatus.setTextColor(Color.RED);
            }
        }, ContextCompat.getMainExecutor(this));
    }
    
    private void processFrame(ImageProxy imageProxy) {
        if (!isAnalyzing) {
            imageProxy.close();
            return;
        }
        
        try {
            Image image = imageProxy.getImage();
            if (image == null) {
                return;
            }
            
            // Convert YUV to RGB bytes
            byte[] rgbBytes = yuvToRgb(image);
            
            if (rgbBytes != null && vitalsModule != null) {
                // Send to Python for processing
                // Note: Downsampling for performance
                byte[] downsampled = downsampleRgb(rgbBytes, 
                        image.getWidth(), image.getHeight(),
                        TARGET_WIDTH, TARGET_HEIGHT);
                
                long timestamp = System.currentTimeMillis();
                vitalsModule.callAttr("add_frame", downsampled, 
                        TARGET_WIDTH, TARGET_HEIGHT, timestamp);
            }
            
        } catch (Exception e) {
            Log.e(TAG, "Frame processing error", e);
        } finally {
            imageProxy.close();
        }
    }
    
    private byte[] yuvToRgb(Image image) {
        try {
            Image.Plane[] planes = image.getPlanes();
            ByteBuffer yBuffer = planes[0].getBuffer();
            ByteBuffer uBuffer = planes[1].getBuffer();
            ByteBuffer vBuffer = planes[2].getBuffer();
            
            int width = image.getWidth();
            int height = image.getHeight();
            
            int ySize = yBuffer.remaining();
            int uSize = uBuffer.remaining();
            int vSize = vBuffer.remaining();
            
            // NV21 format
            byte[] nv21 = new byte[ySize + uSize + vSize];
            yBuffer.get(nv21, 0, ySize);
            vBuffer.get(nv21, ySize, vSize);
            uBuffer.get(nv21, ySize + vSize, uSize);
            
            // Convert NV21 to RGB
            return nv21ToRgb(nv21, width, height);
            
        } catch (Exception e) {
            Log.e(TAG, "YUV conversion error", e);
            return null;
        }
    }
    
    private byte[] nv21ToRgb(byte[] nv21, int width, int height) {
        // Simple NV21 to RGB conversion
        int frameSize = width * height;
        byte[] rgb = new byte[frameSize * 3];
        
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                int y = (0xff & nv21[j * width + i]) - 16;
                int v = (0xff & nv21[frameSize + (j >> 1) * width + (i & ~1)]) - 128;
                int u = (0xff & nv21[frameSize + (j >> 1) * width + (i & ~1) + 1]) - 128;
                
                y = Math.max(0, Math.min(y, 255));
                
                int r = (int) (y + 1.370705f * v);
                int g = (int) (y - 0.698001f * v - 0.337633f * u);
                int b = (int) (y + 1.732446f * u);
                
                r = Math.max(0, Math.min(r, 255));
                g = Math.max(0, Math.min(g, 255));
                b = Math.max(0, Math.min(b, 255));
                
                int idx = (j * width + i) * 3;
                rgb[idx] = (byte) r;
                rgb[idx + 1] = (byte) g;
                rgb[idx + 2] = (byte) b;
            }
        }
        
        return rgb;
    }
    
    private byte[] downsampleRgb(byte[] rgb, int srcWidth, int srcHeight, 
                                  int dstWidth, int dstHeight) {
        byte[] downsampled = new byte[dstWidth * dstHeight * 3];
        
        float xRatio = (float) srcWidth / dstWidth;
        float yRatio = (float) srcHeight / dstHeight;
        
        for (int y = 0; y < dstHeight; y++) {
            for (int x = 0; x < dstWidth; x++) {
                int srcX = (int) (x * xRatio);
                int srcY = (int) (y * yRatio);
                
                int srcIdx = (srcY * srcWidth + srcX) * 3;
                int dstIdx = (y * dstWidth + x) * 3;
                
                downsampled[dstIdx] = rgb[srcIdx];
                downsampled[dstIdx + 1] = rgb[srcIdx + 1];
                downsampled[dstIdx + 2] = rgb[srcIdx + 2];
            }
        }
        
        return downsampled;
    }
    
    private void startMeasurement() {
        if (vitalsModule == null) {
            Toast.makeText(this, "Python not initialized", Toast.LENGTH_SHORT).show();
            return;
        }
        
        isAnalyzing = true;
        updateUIState(true);
        
        // Clear previous data
        vitalsModule.callAttr("clear");
        
        // Start periodic analysis
        mainHandler.postDelayed(analysisRunnable, ANALYSIS_INTERVAL_MS);
        
        tvStatus.setText("Analyzing... Keep still");
        tvStatus.setTextColor(Color.BLUE);
        
        Log.i(TAG, "Measurement started");
    }
    
    private void stopMeasurement() {
        isAnalyzing = false;
        mainHandler.removeCallbacks(analysisRunnable);
        
        if (vitalsModule != null) {
            vitalsModule.callAttr("clear");
        }
        
        updateUIState(false);
        tvStatus.setText("Stopped - Press Start to begin");
        tvStatus.setTextColor(Color.GRAY);
        
        Log.i(TAG, "Measurement stopped");
    }
    
    private void analyzeAndUpdate() {
        try {
            PyObject result = vitalsModule.callAttr("get_vitals");
            String jsonStr = result.toString();
            
            parseAndDisplayResults(jsonStr);
            
        } catch (Exception e) {
            Log.e(TAG, "Analysis error", e);
            tvStatus.setText("Analysis error");
            tvStatus.setTextColor(Color.RED);
        }
    }
    
    private void parseAndDisplayResults(String jsonStr) {
        try {
            Gson gson = new Gson();
            JsonObject result = gson.fromJson(jsonStr, JsonObject.class);
            
            boolean success = result.get("success").getAsBoolean();
            
            if (success && result.has("data")) {
                JsonObject data = result.getAsJsonObject("data");
                
                // Heart Rate
                if (data.has("heart_rate")) {
                    JsonObject hr = data.getAsJsonObject("heart_rate");
                    double bpm = hr.get("bpm").getAsDouble();
                    double confidence = hr.get("confidence").getAsDouble();
                    double hrv = hr.get("hrv_sdnn").getAsDouble();
                    
                    tvHeartRate.setText(String.format("%.0f BPM", bpm));
                    tvHRV.setText(String.format("HRV: %.1f ms", hrv));
                    
                    // Color code based on HR
                    if (bpm < 50 || bpm > 120) {
                        tvHeartRate.setTextColor(Color.RED);
                    } else {
                        tvHeartRate.setTextColor(Color.GREEN);
                    }
                }
                
                // SpO2
                if (data.has("spo2") && !data.get("spo2").isJsonNull()) {
                    JsonObject spo2 = data.getAsJsonObject("spo2");
                    double value = spo2.get("spo2").getAsDouble();
                    tvSpO2.setText(String.format("%.0f%%", value));
                    
                    if (value < 95) {
                        tvSpO2.setTextColor(Color.RED);
                    } else {
                        tvSpO2.setTextColor(Color.GREEN);
                    }
                }
                
                // Respiration
                if (data.has("respiration") && !data.get("respiration").isJsonNull()) {
                    JsonObject resp = data.getAsJsonObject("respiration");
                    double rpm = resp.get("rpm").getAsDouble();
                    tvRespiration.setText(String.format("%.0f RPM", rpm));
                }
                
                // Signal Quality
                if (data.has("signal_quality")) {
                    double quality = data.get("signal_quality").getAsDouble();
                    tvSignalQuality.setText(String.format("Quality: %.0f%%", quality * 100));
                    
                    // Update quality indicator color
                    if (quality > 0.7) {
                        qualityIndicator.setBackgroundColor(Color.GREEN);
                        tvStatus.setText("Good signal - Keep still");
                        tvStatus.setTextColor(Color.GREEN);
                    } else if (quality > 0.4) {
                        qualityIndicator.setBackgroundColor(Color.YELLOW);
                        tvStatus.setText("Fair signal - Adjust position");
                        tvStatus.setTextColor(Color.YELLOW);
                    } else {
                        qualityIndicator.setBackgroundColor(Color.RED);
                        tvStatus.setText("Poor signal - Move to better lighting");
                        tvStatus.setTextColor(Color.RED);
                    }
                }
                
                // Debug info
                if (data.has("timestamp")) {
                    tvDebugInfo.setText("Processing...");
                }
                
            } else {
                // No results yet
                String error = result.has("error") ? result.get("error").getAsString() : "No data";
                tvDebugInfo.setText(error);
                
                if (result.has("buffer_size")) {
                    int bufferSize = result.get("buffer_size").getAsInt();
                    tvStatus.setText(String.format("Collecting data... %d frames", bufferSize));
                }
            }
            
        } catch (Exception e) {
            Log.e(TAG, "Parse error", e);
            tvDebugInfo.setText("Parse error");
        }
    }
    
    private void updateUIState(boolean analyzing) {
        btnStart.setEnabled(!analyzing);
        btnStop.setEnabled(analyzing);
        progressBar.setVisibility(analyzing ? View.VISIBLE : View.GONE);
        
        if (!analyzing) {
            tvHeartRate.setText("-- BPM");
            tvSpO2.setText("--%");
            tvRespiration.setText("-- RPM");
            tvHRV.setText("HRV: --");
            tvSignalQuality.setText("Quality: --");
            tvHeartRate.setTextColor(Color.GRAY);
            tvSpO2.setTextColor(Color.GRAY);
            qualityIndicator.setBackgroundColor(Color.GRAY);
        }
    }
    
    @Override
    protected void onPause() {
        super.onPause();
        stopMeasurement();
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopMeasurement();
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
    }
}
