# FacePhysio Setup Guide

Complete step-by-step guide to build and deploy the pyVHR-powered Android app.

---

## Prerequisites

### Required Software

| Software | Version | Download |
|----------|---------|----------|
| Android Studio | Giraffe (2022.3.1) or newer | https://developer.android.com/studio |
| Python | 3.10.x | https://www.python.org/downloads/ |
| Android SDK | API 24-34 | Via Android Studio |
| JDK | 17 | Bundled with Android Studio |

### Hardware Requirements

- **Development:** 8GB RAM minimum, 16GB recommended
- **Target Device:** Android phone with front camera
  - Android 7.0+ (API 24)
  - 2GB RAM minimum
  - Front-facing camera required

---

## Step 1: Project Setup

### 1.1 Clone/Extract Project

```bash
# Navigate to the implementation folder
cd "/Users/dicksonng/DT/Development/Beta_5.5/Vital sign addon/implementation"
```

### 1.2 Verify Project Structure

```
implementation/
├── app/
│   ├── build.gradle              ✓ App config with Chaquopy
│   ├── proguard-rules.pro        ✓ ProGuard rules
│   └── src/main/
│       ├── AndroidManifest.xml   ✓ App manifest
│       ├── java/com/facephysio/
│       │   └── MainActivity.java ✓ Android UI code
│       ├── python/
│       │   └── vitals_processor.py ✓ Python signal processing
│       └── res/
│           ├── layout/
│           │   └── activity_main.xml ✓ UI layout
│           ├── drawable/
│           │   └── circle_gray.xml   ✓ Quality indicator
│           └── values/
│               └── strings.xml   ✓ String resources
├── build.gradle                  ✓ Project config
└── settings.gradle               ✓ Settings
```

---

## Step 2: Open in Android Studio

### 2.1 Import Project

1. Open Android Studio
2. Click **"Open"** (not "New Project")
3. Navigate to `implementation` folder
4. Select and click **OK**

### 2.2 First Sync (IMPORTANT)

**First sync will take 5-10 minutes.** Chaquopy downloads:
- Python 3.10 runtime (~50MB)
- NumPy pre-built for Android (~15MB)
- SciPy pre-built for Android (~30MB)

**Monitor progress:**
```
Build → Build Output (bottom panel)
```

Look for:
```
> Configure project :app
> Chaquopy: Installing Python runtime
> Chaquopy: Installing numpy
> Chaquopy: Installing scipy
```

---

## Step 3: Configure Python Path (If Needed)

If you get "Python not found" error:

### Windows
```gradle
// app/build.gradle
python {
    buildPython "C:/Python310/python.exe"
}
```

### macOS/Linux
```gradle
// app/build.gradle  
python {
    buildPython "/usr/local/bin/python3.10"
}
```

### Verify Python Version

```bash
python3.10 --version
# Should show: Python 3.10.x
```

---

## Step 4: Build Project

### 4.1 Clean Build

```bash
# Terminal in Android Studio
./gradlew clean
```

### 4.2 Build Debug APK

```bash
./gradlew assembleDebug
```

**Or use Android Studio:**
```
Build → Build Bundle(s) / APK(s) → Build APK(s)
```

### 4.3 Verify Build Output

APK should be created at:
```
app/build/outputs/apk/debug/app-debug.apk
```

Size should be ~40-60MB (includes Python runtime).

---

## Step 5: Deploy to Device

### 5.1 Enable Developer Options on Phone

1. Go to **Settings → About Phone**
2. Tap **Build Number** 7 times
3. Enter PIN if prompted
4. Developer options enabled!

### 5.2 Enable USB Debugging

1. Go to **Settings → System → Developer Options**
2. Enable **USB Debugging**
3. Connect phone to computer via USB
4. Accept "Allow USB debugging?" prompt on phone

### 5.3 Install via ADB

```bash
# Verify device is connected
adb devices

# Install APK
adb install app/build/outputs/apk/debug/app-debug.apk
```

**Or use Android Studio:**
```
Run → Run 'app' (or click green play button)
```

### 5.4 Verify Installation

App icon: **FacePhysio** should appear on phone.

---

## Step 6: First Run

### 6.1 Grant Permissions

On first launch, app will request:
- **Camera permission** - Required for rPPG

Tap **"Allow"**

### 6.2 Test Basic Functionality

1. **Launch app** - Camera preview should appear
2. **Position face** - Center in preview, ensure forehead visible
3. **Tap "Start"** - Analysis begins
4. **Wait 5-10 seconds** - For buffer to fill
5. **Check results** - Heart rate should appear

### 6.3 Check Signal Quality

| Indicator | Color | Meaning | Action |
|-----------|-------|---------|--------|
| 🔵 | Gray | Initializing | Wait |
| 🟢 | Green | Good signal (>70%) | Keep still |
| 🟡 | Yellow | Fair signal (40-70%) | Adjust position |
| 🔴 | Red | Poor signal (<40%) | Improve lighting |

---

## Troubleshooting

### Build Errors

#### "Could not find com.chaquo.python:gradle:14.0.2"

**Solution:** Add Chaquopy repository

```gradle
// settings.gradle
pluginManagement {
    repositories {
        maven { url "https://chaquo.com/maven" }
    }
}
```

#### "No compatible Python runtime found"

**Solution:** Check ABI filters

```gradle
// app/build.gradle
android {
    defaultConfig {
        ndk {
            abiFilters "arm64-v8a", "armeabi-v7a"
        }
    }
}
```

#### "Out of memory" during build

**Solution:** Increase Gradle heap size

```gradle
// gradle.properties
org.gradle.jvmargs=-Xmx4g
```

### Runtime Errors

#### "Python not initialized"

**Check logcat:**
```bash
adb logcat -s FacePhysio:D
```

**Common causes:**
- Chaquopy license expired (development mode works)
- Python files not in `src/main/python`
- Build didn't complete successfully

#### "No face detected"

**Solutions:**
- Center face in preview
- Ensure good lighting (avoid backlight)
- Remove glasses/sunglasses
- Move to area with even lighting

#### "Poor signal quality"

**Solutions:**
- Stay completely still
- Face camera directly
- Improve lighting conditions
- Move away from flickering lights
- Clean camera lens

#### App crashes on start

**Check:**
```bash
adb logcat -s FacePhysio:E
```

**Common fixes:**
1. Clean and rebuild
2. Uninstall and reinstall
3. Check device has enough RAM (2GB+)
4. Verify Android version (7.0+)

---

## Performance Optimization

### Reduce APK Size

```gradle
// app/build.gradle
android {
    defaultConfig {
        ndk {
            // Only arm64 (modern phones)
            abiFilters "arm64-v8a"
        }
    }
}
```

### Improve Processing Speed

In `vitals_processor.py`:
```python
# Use simpler algorithm (faster, slightly less accurate)
ppg_data = self.extract_ppg_signal(use_green_only=True)

# Or reduce buffer size
_processor = VitalSignsProcessor(fps=30, buffer_seconds=3)  # 3 sec instead of 5
```

### Reduce Battery Usage

In `MainActivity.java`:
```java
// Analyze less frequently
private static final int ANALYSIS_INTERVAL_MS = 3000; // 3 seconds instead of 2
```

---

## Advanced Configuration

### Customize Vital Sign Thresholds

```python
# vitals_processor.py
class VitalSignsProcessor:
    def __init__(self):
        self.min_bpm = 40       # Lower HR threshold
        self.max_bpm = 200      # Upper HR threshold
        self.min_spo2 = 85      # Lower SpO2 threshold
        self.min_snr_db = 6.0   # Lower SNR requirement
```

### Add Data Export

```java
// MainActivity.java
private void exportToCSV() {
    // Implementation for data export
    // See commented code in original file
}
```

### Integrate with Backend

```java
// Add to analysis results
private void sendToCloud(JsonObject results) {
    // HTTP POST to your backend
    // Include timestamp, device ID, vitals
}
```

---

## Testing Checklist

### Functional Tests

- [ ] App launches without crash
- [ ] Camera preview displays
- [ ] Face detection works
- [ ] Start/Stop buttons function
- [ ] Heart rate calculates (10+ seconds)
- [ ] Signal quality indicator updates
- [ ] Results display updates

### Edge Cases

- [ ] No face in frame
- [ ] Multiple faces in frame
- [ ] Rapid movement
- [ ] Poor lighting (dark room)
- [ ] Backlight (window behind)
- [ ] Phone rotation
- [ ] Background apps running

### Performance Tests

- [ ] 5-minute continuous monitoring
- [ ] Battery usage <10% per hour
- [ ] Memory usage <200MB
- [ ] No memory leaks (check with Profiler)

---

## Next Steps

After successful setup:

1. **Calibration Study**
   - Compare against FDA-cleared pulse oximeter
   - Test on 20+ subjects
   - Document accuracy metrics

2. **Clinical Validation**
   - IRB approval if for research
   - Informed consent forms
   - Data collection protocols

3. **Production Hardening**
   - Chaquopy commercial license
   - Error handling improvements
   - Cloud backend integration
   - Data encryption

4. **Regulatory Compliance**
   - FDA 510(k) preparation (if medical device)
   - HIPAA compliance (if handling PHI)
   - CE marking (if EU market)

---

## Support Resources

- **Chaquopy Docs:** https://chaquo.com/chaquopy/doc/current/
- **pyVHR Paper:** https://arxiv.org/abs/2104.04407
- **CameraX Guide:** https://developer.android.com/training/camerax
- **rPPG Overview:** https://arxiv.org/abs/2304.12909

---

**Last Updated:** 2026-02-03  
**Version:** 1.0
