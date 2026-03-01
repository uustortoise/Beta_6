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
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.facephysio.data.AppDatabase;
import com.facephysio.data.Measurement;
import com.facephysio.data.MeasurementDao;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

/**
 * Main Activity for FacePhysio Vital Signs Monitor - Pilot Version
 * 
 * Features:
 * - Real-time vital sign monitoring (HR, SpO2, Respiration)
 * - Motion artifact detection with pause/resume
 * - Population-based skin tone calibration
 * - Room database persistence
 * - Session management
 * - Data export to CSV
 */
public class MainActivity extends AppCompatActivity {
    
    private static final String TAG = "FacePhysio";
    private static final int CAMERA_PERMISSION_REQUEST = 100;
    private static final int ANALYSIS_INTERVAL_MS = 2000;
    
    // UI Components
    private PreviewView previewView;
    private TextView tvStatus;
    private TextView tvHeartRate;
    private TextView tvSpO2;
    private TextView tvRespiration;
    private TextView tvHRV;
    private TextView tvSignalQuality;
    private TextView tvMotionStatus;
    private TextView tvSkinTone;
    private TextView tvSessionInfo;
    private Button btnStart;
    private Button btnStop;
    private Button btnExport;
    private Button btnHistory;
    private ProgressBar progressBar;
    private View qualityIndicator;
    private View motionOverlay;
    
    // Camera
    private ProcessCameraProvider cameraProvider;
    private final Executor cameraExecutor = Executors.newSingleThreadExecutor();
    private boolean isAnalyzing = false;
    
    // Python/Chaquopy
    private Python py;
    private PyObject vitalsModule;
    
    // Database
    private AppDatabase database;
    private MeasurementDao measurementDao;
    
    // Session
    private String currentSessionId;
    private int sessionMeasurementCount = 0;
    private long sessionStartTime;
    
    // Handler
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private Runnable analysisRunnable;
    
    // Frame dimensions
    private int frameWidth = 640;
    private int frameHeight = 480;
    private static final int TARGET_WIDTH = 320;
    private static final int TARGET_HEIGHT = 240;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        initializeViews();
        initializePython();
        initializeDatabase();
        setupButtons();
        setupAnalysisRunnable();
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
        tvMotionStatus = findViewById(R.id.tvMotionStatus);
        tvSkinTone = findViewById(R.id.tvSkinTone);
        tvSessionInfo = findViewById(R.id.tvSessionInfo);
        btnStart = findViewById(R.id.btnStart);
        btnStop = findViewById(R.id.btnStop);
        btnExport = findViewById(R.id.btnExport);
        btnHistory = findViewById(R.id.btnHistory);
        progressBar = findViewById(R.id.progressBar);
        qualityIndicator = findViewById(R.id.qualityIndicator);
        motionOverlay = findViewById(R.id.motionOverlay);
        
        updateUIState(false);
    }
    
    private void initializePython() {
        try {
            if (!Python.isStarted()) {
                Python.start(new AndroidPlatform(this));
            }
            py = Python.getInstance();
            vitalsModule = py.getModule("vitals_processor");
            vitalsModule.callAttr("initialize", 30, 5);
            Log.i(TAG, "Python initialized");
        } catch (Exception e) {
            Log.e(TAG, "Python init failed", e);
            tvStatus.setText("Error: Python init failed");
            tvStatus.setTextColor(Color.RED);
        }
    }
    
    private void initializeDatabase() {
        database = AppDatabase.getInstance(this);
        measurementDao = database.measurementDao();
        Log.i(TAG, "Database initialized");
    }
    
    private void setupButtons() {
        btnStart.setOnClickListener(v -> startSession());
        btnStop.setOnClickListener(v -> endSession());
        btnExport.setOnClickListener(v -> exportData());
        btnHistory.setOnClickListener(v -> showHistory());
    }
    
    private void setupAnalysisRunnable() {
        analysisRunnable = () -> {
            if (isAnalyzing) {
                analyzeAndUpdate();
                mainHandler.postDelayed(analysisRunnable, ANALYSIS_INTERVAL_MS);
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
                
                Preview preview = new Preview.Builder()
                        .setTargetResolution(new android.util.Size(frameWidth, frameHeight))
                        .build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());
                
                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setTargetResolution(new android.util.Size(frameWidth, frameHeight))
                        .build();
                
                imageAnalysis.setAnalyzer(cameraExecutor, this::processCameraFrame);
                
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                        .build();
                
                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
                
                tvStatus.setText("Camera ready - Press Start");
                tvStatus.setTextColor(Color.GREEN);
                
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Camera failed", e);
                tvStatus.setText("Camera error");
                tvStatus.setTextColor(Color.RED);
            }
        }, ContextCompat.getMainExecutor(this));
    }
    
    private void processCameraFrame(ImageProxy imageProxy) {
        if (!isAnalyzing) {
            imageProxy.close();
            return;
        }
        
        Image image = imageProxy.getImage();
        if (image == null) {
            imageProxy.close();
            return;
        }
        
        try {
            byte[] rgbBytes = yuvToRgb(image);
            if (rgbBytes != null && vitalsModule != null) {
                byte[] downsampled = downsampleRgb(rgbBytes, 
                        image.getWidth(), image.getHeight(),
                        TARGET_WIDTH, TARGET_HEIGHT);
                
                long timestamp = System.currentTimeMillis();
                PyObject result = vitalsModule.callAttr("add_frame", downsampled, 
                        TARGET_WIDTH, TARGET_HEIGHT, timestamp);
                
                // Parse motion detection result
                String jsonStr = result.toString();
                updateMotionUI(jsonStr);
            }
        } catch (Exception e) {
            Log.e(TAG, "Frame processing error", e);
        } finally {
            imageProxy.close();
        }
    }
    
    private void updateMotionUI(String jsonStr) {
        try {
            Gson gson = new Gson();
            JsonObject result = gson.fromJson(jsonStr, JsonObject.class);
            
            if (result.has("motion_detected") && result.has("motion_cooldown")) {
                boolean motionDetected = result.get("motion_detected").getAsBoolean();
                int cooldown = result.get("motion_cooldown").getAsInt();
                double motionScore = result.get("motion_score").getAsDouble();
                
                runOnUiThread(() -> {
                    if (cooldown > 0 || motionDetected) {
                        // Show motion overlay
                        motionOverlay.setVisibility(View.VISIBLE);
                        motionOverlay.setAlpha(0.7f);
                        tvMotionStatus.setText(String.format("Motion detected! Pausing... (%.1f)", motionScore));
                        tvMotionStatus.setTextColor(Color.RED);
                        tvStatus.setText("Hold still to resume");
                        tvStatus.setTextColor(Color.YELLOW);
                    } else {
                        // Hide motion overlay
                        motionOverlay.setVisibility(View.GONE);
                        tvMotionStatus.setText("Stable");
                        tvMotionStatus.setTextColor(Color.GREEN);
                    }
                });
            }
        } catch (Exception e) {
            Log.e(TAG, "Motion UI update error", e);
        }
    }
    
    private void startSession() {
        if (vitalsModule == null) {
            Toast.makeText(this, "Python not initialized", Toast.LENGTH_SHORT).show();
            return;
        }
        
        // Start new session in Python
        PyObject sessionResult = vitalsModule.callAttr("start_session");
        currentSessionId = sessionResult.toString();
        sessionMeasurementCount = 0;
        sessionStartTime = System.currentTimeMillis();
        
        isAnalyzing = true;
        updateUIState(true);
        
        vitalsModule.callAttr("clear");
        
        mainHandler.postDelayed(analysisRunnable, ANALYSIS_INTERVAL_MS);
        
        tvStatus.setText("Session started - Analyzing...");
        tvStatus.setTextColor(Color.BLUE);
        tvSessionInfo.setText("Session: " + currentSessionId.substring(0, 8));
        
        Log.i(TAG, "Session started: " + currentSessionId);
    }
    
    private void endSession() {
        isAnalyzing = false;
        mainHandler.removeCallbacks(analysisRunnable);
        
        // End session in Python and get summary
        if (vitalsModule != null) {
            try {
                PyObject summaryResult = vitalsModule.callAttr("end_session");
                Log.i(TAG, "Session ended: " + summaryResult.toString());
            } catch (Exception e) {
                Log.e(TAG, "Error ending session", e);
            }
            vitalsModule.callAttr("clear");
        }
        
        updateUIState(false);
        tvStatus.setText("Session ended");
        tvStatus.setTextColor(Color.GRAY);
        motionOverlay.setVisibility(View.GONE);
        
        // Show summary
        long duration = (System.currentTimeMillis() - sessionStartTime) / 1000;
        Toast.makeText(this, 
                String.format("Session complete!\nDuration: %d sec\nMeasurements: %d", 
                        duration, sessionMeasurementCount),
                Toast.LENGTH_LONG).show();
        
        currentSessionId = null;
    }
    
    private void analyzeAndUpdate() {
        try {
            PyObject result = vitalsModule.callAttr("get_vitals");
            String jsonStr = result.toString();
            
            parseAndDisplayResults(jsonStr);
            
        } catch (Exception e) {
            Log.e(TAG, "Analysis error", e);
        }
    }
    
    private void parseAndDisplayResults(String jsonStr) {
        try {
            Gson gson = new Gson();
            JsonObject result = gson.fromJson(jsonStr, JsonObject.class);
            
            boolean success = result.get("success").getAsBoolean();
            
            if (success && result.has("data")) {
                JsonObject data = result.getAsJsonObject("data");
                
                // Save to database
                saveMeasurement(data);
                
                // Update UI
                runOnUiThread(() -> {
                    updateVitalsUI(data);
                });
                
            } else {
                String error = result.has("error") ? result.get("error").getAsString() : "No data";
                runOnUiThread(() -> {
                    if (result.has("buffer_size")) {
                        int bufferSize = result.get("buffer_size").getAsInt();
                        tvStatus.setText(String.format("Collecting data... %d frames", bufferSize));
                    }
                });
            }
            
        } catch (Exception e) {
            Log.e(TAG, "Parse error", e);
        }
    }
    
    private void saveMeasurement(JsonObject data) {
        try {
            Measurement m = new Measurement();
            m.setSessionUuid(currentSessionId);
            m.setTimestamp(System.currentTimeMillis());
            m.setMeasurementId(++sessionMeasurementCount);
            
            // Device info
            m.setDeviceModel(android.os.Build.MODEL);
            m.setAndroidVersion(android.os.Build.VERSION.RELEASE);
            m.setAppVersion("1.0.0");
            
            // Heart rate
            if (data.has("heart_rate")) {
                JsonObject hr = data.getAsJsonObject("heart_rate");
                m.setHeartRate(hr.get("bpm").getAsDouble());
                m.setHeartRateConfidence(hr.get("confidence").getAsDouble());
                m.setHrvSdnn(hr.has("hrv_sdnn") ? hr.get("hrv_sdnn").getAsDouble() : null);
            }
            
            // SpO2
            if (data.has("spo2") && !data.get("spo2").isJsonNull()) {
                JsonObject spo2 = data.getAsJsonObject("spo2");
                m.setSpo2(spo2.get("spo2").getAsDouble());
                m.setSpo2Confidence(spo2.get("confidence").getAsDouble());
            }
            
            // Respiration
            if (data.has("respiration") && !data.get("respiration").isJsonNull()) {
                JsonObject resp = data.getAsJsonObject("respiration");
                m.setRespirationRate(resp.get("rpm").getAsDouble());
                m.setRespirationConfidence(resp.get("confidence").getAsDouble());
            }
            
            // Quality and motion
            if (data.has("signal_quality")) {
                m.setSignalQuality(data.get("signal_quality").getAsDouble());
            }
            
            if (data.has("motion_state")) {
                JsonObject motion = data.getAsJsonObject("motion_state");
                m.setInMotionCooldown(motion.get("in_cooldown").getAsBoolean());
            }
            
            // Skin tone
            if (data.has("skin_tone")) {
                JsonObject skin = data.getAsJsonObject("skin_tone");
                m.setSkinToneCategory(skin.get("category").getAsString());
                m.setSkinToneValue(skin.get("value").getAsDouble());
            }
            
            // Calibration
            if (data.has("heart_rate") && data.getAsJsonObject("heart_rate").has("calibration_applied")) {
                m.setCalibrationApplied(data.getAsJsonObject("heart_rate").get("calibration_applied").getAsString());
            }
            
            // Save asynchronously
            new Thread(() -> {
                measurementDao.insert(m);
                Log.d(TAG, "Saved measurement " + sessionMeasurementCount);
            }).start();
            
        } catch (Exception e) {
            Log.e(TAG, "Save measurement error", e);
        }
    }
    
    private void updateVitalsUI(JsonObject data) {
        // Heart Rate
        if (data.has("heart_rate")) {
            JsonObject hr = data.getAsJsonObject("heart_rate");
            double bpm = hr.get("bpm").getAsDouble();
            double confidence = hr.get("confidence").getAsDouble();
            tvHeartRate.setText(String.format(Locale.getDefault(), "%.0f BPM", bpm));
            tvHRV.setText(String.format("HRV: %.1f ms", hr.get("hrv_sdnn").getAsDouble()));
            
            if (bpm < 50 || bpm > 120) {
                tvHeartRate.setTextColor(Color.YELLOW);
            } else {
                tvHeartRate.setTextColor(Color.GREEN);
            }
        }
        
        // SpO2
        if (data.has("spo2") && !data.get("spo2").isJsonNull()) {
            JsonObject spo2 = data.getAsJsonObject("spo2");
            double value = spo2.get("spo2").getAsDouble();
            tvSpO2.setText(String.format(Locale.getDefault(), "%.0f%%", value));
            tvSpO2.setTextColor(value < 95 ? Color.YELLOW : Color.GREEN);
        }
        
        // Respiration
        if (data.has("respiration") && !data.get("respiration").isJsonNull()) {
            JsonObject resp = data.getAsJsonObject("respiration");
            tvRespiration.setText(String.format(Locale.getDefault(), "%.0f RPM", resp.get("rpm").getAsDouble()));
        }
        
        // Signal Quality
        if (data.has("signal_quality")) {
            double quality = data.get("signal_quality").getAsDouble();
            tvSignalQuality.setText(String.format("Quality: %.0f%%", quality * 100));
            
            if (quality > 0.7) {
                qualityIndicator.setBackgroundColor(Color.GREEN);
                tvStatus.setText("Good signal");
                tvStatus.setTextColor(Color.GREEN);
            } else if (quality > 0.4) {
                qualityIndicator.setBackgroundColor(Color.YELLOW);
                tvStatus.setText("Fair signal - Adjust position");
                tvStatus.setTextColor(Color.YELLOW);
            } else {
                qualityIndicator.setBackgroundColor(Color.RED);
                tvStatus.setText("Poor signal - Check lighting");
                tvStatus.setTextColor(Color.RED);
            }
        }
        
        // Skin Tone
        if (data.has("skin_tone")) {
            JsonObject skin = data.getAsJsonObject("skin_tone");
            String category = skin.get("category").getAsString();
            tvSkinTone.setText("Skin: " + category);
        }
        
        // Session info
        tvSessionInfo.setText(String.format("Session: %s | Reading: %d", 
                currentSessionId.substring(0, 8), sessionMeasurementCount));
    }
    
    private void updateUIState(boolean analyzing) {
        btnStart.setEnabled(!analyzing);
        btnStop.setEnabled(analyzing);
        btnExport.setEnabled(!analyzing);
        btnHistory.setEnabled(!analyzing);
        progressBar.setVisibility(analyzing ? View.VISIBLE : View.GONE);
        
        if (!analyzing) {
            resetDisplays();
        }
    }
    
    private void resetDisplays() {
        tvHeartRate.setText("-- BPM");
        tvSpO2.setText("--%");
        tvRespiration.setText("-- RPM");
        tvHRV.setText("HRV: --");
        tvSignalQuality.setText("Quality: --");
        tvMotionStatus.setText("--");
        tvSkinTone.setText("Skin: --");
        tvSessionInfo.setText("No active session");
        tvHeartRate.setTextColor(Color.GRAY);
        tvSpO2.setTextColor(Color.GRAY);
        qualityIndicator.setBackgroundColor(Color.GRAY);
        motionOverlay.setVisibility(View.GONE);
    }
    
    private void exportData() {
        new Thread(() -> {
            try {
                List<Measurement> measurements = measurementDao.getAllMeasurements();
                if (measurements.isEmpty()) {
                    runOnUiThread(() -> Toast.makeText(this, "No data to export", Toast.LENGTH_SHORT).show());
                    return;
                }
                
                File exportDir = new File(getExternalFilesDir(null), "exports");
                if (!exportDir.exists()) {
                    exportDir.mkdirs();
                }
                
                String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
                File csvFile = new File(exportDir, "facephysio_export_" + timestamp + ".csv");
                
                FileWriter writer = new FileWriter(csvFile);
                
                // Header
                writer.write("id,session_uuid,timestamp,measurement_id,heart_rate,heart_rate_confidence," +
                        "hrv_sdnn,spo2,spo2_confidence,respiration_rate,respiration_confidence," +
                        "signal_quality,skin_tone_category,skin_tone_value,calibration_applied," +
                        "device_model,android_version,app_version\n");
                
                // Data
                for (Measurement m : measurements) {
                    writer.write(String.format(Locale.US,
                            "%d,%s,%d,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.2f,%s,%.3f,%s,%s,%s,%s\n",
                            m.getId(),
                            m.getSessionUuid(),
                            m.getTimestamp(),
                            m.getMeasurementId(),
                            m.getHeartRate() != null ? m.getHeartRate() : 0,
                            m.getHeartRateConfidence() != null ? m.getHeartRateConfidence() : 0,
                            m.getHrvSdnn() != null ? m.getHrvSdnn() : 0,
                            m.getSpo2() != null ? m.getSpo2() : 0,
                            m.getSpo2Confidence() != null ? m.getSpo2Confidence() : 0,
                            m.getRespirationRate() != null ? m.getRespirationRate() : 0,
                            m.getRespirationConfidence() != null ? m.getRespirationConfidence() : 0,
                            m.getSignalQuality() != null ? m.getSignalQuality() : 0,
                            m.getSkinToneCategory() != null ? m.getSkinToneCategory() : "",
                            m.getSkinToneValue() != null ? m.getSkinToneValue() : 0,
                            m.getCalibrationApplied() != null ? m.getCalibrationApplied() : "",
                            m.getDeviceModel(),
                            m.getAndroidVersion(),
                            m.getAppVersion()
                    ));
                }
                
                writer.close();
                
                runOnUiThread(() -> Toast.makeText(this, 
                        "Exported " + measurements.size() + " records to:\n" + csvFile.getAbsolutePath(),
                        Toast.LENGTH_LONG).show());
                
            } catch (IOException e) {
                Log.e(TAG, "Export error", e);
                runOnUiThread(() -> Toast.makeText(this, "Export failed", Toast.LENGTH_SHORT).show());
            }
        }).start();
    }
    
    private void showHistory() {
        new Thread(() -> {
            int count = measurementDao.getMeasurementCount();
            List<String> sessions = measurementDao.getAllSessionIds();
            
            runOnUiThread(() -> {
                StringBuilder msg = new StringBuilder();
                msg.append("Total measurements: ").append(count).append("\n\n");
                msg.append("Recent sessions:\n");
                
                int displayed = 0;
                for (String session : sessions) {
                    if (displayed++ >= 5) break;
                    int sessionCount = measurementDao.getSessionMeasurementCount(session);
                    msg.append(session.substring(0, 8)).append("... : ")
                            .append(sessionCount).append(" readings\n");
                }
                
                Toast.makeText(this, msg.toString(), Toast.LENGTH_LONG).show();
            });
        }).start();
    }
    
    // Helper methods (same as before)
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
            
            byte[] nv21 = new byte[ySize + uSize + vSize];
            yBuffer.get(nv21, 0, ySize);
            vBuffer.get(nv21, ySize, vSize);
            uBuffer.get(nv21, ySize + vSize, uSize);
            
            return nv21ToRgb(nv21, width, height);
            
        } catch (Exception e) {
            Log.e(TAG, "YUV conversion error", e);
            return null;
        }
    }
    
    private byte[] nv21ToRgb(byte[] nv21, int width, int height) {
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
    
    @Override
    protected void onPause() {
        super.onPause();
        if (isAnalyzing) {
            endSession();
        }
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (isAnalyzing) {
            endSession();
        }
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        AppDatabase.destroyInstance();
    }
}
