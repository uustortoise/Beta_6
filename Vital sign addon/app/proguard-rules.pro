# ProGuard rules for FacePhysio

# Keep Python interface
-keep class com.chaquo.python.** { *; }
-keep class com.facephysio.BuildConfig { *; }

# Keep CameraX
-keep class androidx.camera.** { *; }

# Keep Gson
-keep class com.google.gson.** { *; }
-keepclassmembers class * {
    @com.google.gson.annotations.SerializedName <fields>;
}

# Keep model classes
-keep class com.facephysio.** { *; }

# Python specific
-keepclassmembers class * {
    native <methods>;
}
