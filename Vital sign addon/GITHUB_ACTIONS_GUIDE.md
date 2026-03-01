# GitHub Actions Build Guide

This guide explains how to use GitHub Actions to automatically build the FacePhysio Android APK.

---

## 🤔 What is GitHub Actions?

**GitHub Actions** is a free CI/CD (Continuous Integration/Continuous Deployment) service that:
- Automatically builds your code when you push changes
- Runs on GitHub's servers (not your computer)
- Can create APK files without needing Android Studio
- Uploads build artifacts for download

**Best part:** It's **FREE** for public repositories!

---

## 📁 Files Created

```
Vital sign addon/
├── .github/
│   └── workflows/
│       └── build-android.yml    # GitHub Actions workflow
├── implementation/
│   ├── gradlew                   # Gradle wrapper script
│   └── gradle/wrapper/
│       ├── gradle-wrapper.properties
│       └── gradle-wrapper.jar
└── GITHUB_ACTIONS_GUIDE.md       # This guide
```

---

## 🚀 How to Use

### Step 1: Push to GitHub

First, you need to push this code to a GitHub repository:

```bash
# Navigate to the Vital sign addon folder
cd "Beta_5.5/Vital sign addon"

# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit with GitHub Actions workflow"

# Create GitHub repository (via web or CLI)
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/facephysio.git
git push -u origin main
```

### Step 2: GitHub Actions Automatically Builds

Once pushed, GitHub Actions will:
1. **Trigger** on every push to `main` or `master` branch
2. **Build** the APK using Ubuntu + JDK 17 + Python 3.10
3. **Upload** the APK as a downloadable artifact
4. **Create** a GitHub Release (optional)

### Step 3: Download the APK

**Option A: From GitHub Actions Artifacts**
1. Go to your GitHub repository
2. Click **Actions** tab
3. Click the latest workflow run
4. Scroll to **Artifacts** section
5. Download `FacePhysio-Debug-APK`

**Option B: From GitHub Releases** (auto-created)
1. Go to **Releases** section
2. Download the APK from the latest release

---

## 📋 Workflow Triggers

The workflow runs when:

| Trigger | When It Runs |
|---------|--------------|
| `push` to `main`/`master` | Every code push |
| `pull_request` | When PR is created/updated |
| `workflow_dispatch` | Manual trigger (button in GitHub UI) |

### Manual Trigger

You can manually trigger a build:
1. Go to **Actions** tab
2. Select **Build Android APK**
3. Click **Run workflow**
4. Choose build type (debug/release)
5. Click **Run workflow**

---

## 🔧 Build Process Explained

```
┌─────────────────────────────────────────────────────────────┐
│  GitHub Actions Workflow (Ubuntu Server)                    │
├─────────────────────────────────────────────────────────────┤
│  1. Checkout Code                                           │
│     └─> Get latest code from repository                     │
│                                                             │
│  2. Setup JDK 17                                            │
│     └─> Java Development Kit for Android                    │
│                                                             │
│  3. Setup Python 3.10                                       │
│     └─> Required for Chaquopy (Python in Android)           │
│                                                             │
│  4. Cache Gradle                                            │
│     └─> Speed up future builds                              │
│                                                             │
│  5. Build APK                                               │
│     └─> ./gradlew assembleDebug                             │
│         ├─> Download dependencies                           │
│         ├─> Compile Java code                               │
│         ├─> Process Python with Chaquopy                    │
│         └─> Package into APK                                │
│                                                             │
│  6. Upload Artifact                                         │
│     └─> Store APK for download (30 days)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Build Outputs

| Build Type | Output File | Size |
|------------|-------------|------|
| Debug | `app-debug.apk` | ~15-25 MB |
| Release | `app-release-unsigned.apk` | ~10-20 MB |

---

## ⚙️ Workflow Configuration

### File: `.github/workflows/build-android.yml`

Key sections:

```yaml
# When to run
trigger:
  push:
    branches: [main, master]
  workflow_dispatch:  # Manual trigger

# Build environment
runs-on: ubuntu-latest

# Steps
steps:
  - uses: actions/checkout@v4      # Get code
  - uses: actions/setup-java@v4    # Setup Java 17
  - uses: actions/setup-python@v5  # Setup Python 3.10
  - run: ./gradlew assembleDebug   # Build APK
  - uses: actions/upload-artifact@v4  # Upload APK
```

---

## 🐛 Troubleshooting

### Build Fails

| Issue | Solution |
|-------|----------|
| "Python not found" | Check Python 3.10 setup step |
| "Gradle permission denied" | Make `gradlew` executable |
| "Chaquopy license" | Free for development, contact Chaquopy for production |
| Build timeout | Increase `timeout-minutes` in workflow |

### Check Build Logs

1. Go to **Actions** tab
2. Click failed workflow
3. Click the failed job
4. Expand steps to see error messages

---

## 🔒 Security Notes

- **Debug APKs** are safe for testing but not for Play Store
- **Release APKs** should be signed for distribution
- GitHub Actions uses encrypted secrets for sensitive data
- Artifacts are stored securely for 30 days

---

## 💡 Tips

1. **Add a badge** to your README:
   ```markdown
   ![Build Status](https://github.com/YOUR_USERNAME/facephysio/workflows/Build%20Android%20APK/badge.svg)
   ```

2. **Cache dependencies** to speed up builds (already configured)

3. **Matrix builds** can test multiple Android versions

4. **Notifications** can be sent to Slack/Email on build completion

---

## 📚 Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Android CI/CD Guide](https://developer.android.com/studio/build/building-cmdline)
- [Chaquopy Documentation](https://chaquo.com/chaquopy/doc/current/)

---

## ✅ Quick Checklist

- [ ] Push code to GitHub repository
- [ ] Check Actions tab for workflow runs
- [ ] Download APK from artifacts
- [ ] Test on Android device
- [ ] Share APK with your team

---

**Questions?** Check the GitHub Actions logs or refer to the troubleshooting section above.
