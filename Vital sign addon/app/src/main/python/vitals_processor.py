"""
Vital Signs Processor - Pilot Implementation
Production-ready rPPG processing for Android with:
- Motion artifact detection
- Population-based skin tone calibration
- Session management
- Enhanced signal quality assessment
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from collections import deque
import json
import uuid
from datetime import datetime

class VitalSignsProcessor:
    """
    Production-grade vital signs processor for pilot study.
    """
    
    def __init__(self, fps=30, buffer_seconds=5):
        self.fps = fps
        self.buffer_size = int(fps * buffer_seconds)
        
        # Buffers
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.timestamps = deque(maxlen=self.buffer_size)
        self.prev_frame = None
        self.motion_history = deque(maxlen=10)  # Last 10 motion checks
        
        # ROI tracking
        self.forehead_roi = None
        self.roi_stable_count = 0
        
        # Signal processing parameters
        self.min_bpm = 45
        self.max_bpm = 180
        self.lowcut = self.min_bpm / 60.0
        self.highcut = self.max_bpm / 60.0
        
        # Motion detection
        self.motion_threshold = 15.0  # Mean pixel difference threshold
        self.motion_cooldown = 0
        self.consecutive_motion_frames = 0
        
        # Skin tone calibration
        self.skin_tone_category = None
        self.skin_tone_value = None
        self.calibration_offsets = {
            'light': {'spo2': 0.0, 'hr_gain': 1.0},
            'medium': {'spo2': 1.5, 'hr_gain': 1.05},
            'dark': {'spo2': 3.0, 'hr_gain': 1.10}
        }
        
        # Quality thresholds
        self.min_signal_quality = 0.5
        self.min_snr_db = 6.0
        
        # Session
        self.session_id = None
        self.measurement_count = 0
        
    def start_session(self):
        """Start a new measurement session."""
        self.session_id = str(uuid.uuid4())
        self.clear_buffer()
        self.measurement_count = 0
        return self.session_id
    
    def end_session(self):
        """End current session and return summary."""
        summary = {
            'session_id': self.session_id,
            'total_measurements': self.measurement_count,
            'duration_seconds': len(self.frame_buffer) / self.fps if self.frame_buffer else 0
        }
        self.session_id = None
        return summary
    
    def add_frame(self, frame_bytes, width, height, timestamp=None):
        """Add frame with motion detection."""
        try:
            expected_size = width * height * 3
            if len(frame_bytes) < expected_size:
                return {'success': False, 'error': 'Invalid frame size'}
            
            frame = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = frame[:expected_size].reshape((height, width, 3))
            
            # Motion detection
            motion_detected, motion_score = self._detect_motion(frame)
            
            # Handle motion state
            if motion_detected:
                self.consecutive_motion_frames += 1
                if self.consecutive_motion_frames >= 2:  # Sustained motion
                    self.motion_cooldown = int(self.fps * 1.5)  # 1.5s cooldown
            else:
                self.consecutive_motion_frames = 0
                if self.motion_cooldown > 0:
                    self.motion_cooldown -= 1
            
            # Only add frame if not in motion cooldown
            if self.motion_cooldown == 0:
                self.frame_buffer.append(frame)
                self.timestamps.append(timestamp or int(datetime.now().timestamp() * 1000))
                
                # Detect skin tone on first few frames
                if self.skin_tone_category is None and len(self.frame_buffer) >= 10:
                    self._detect_skin_tone(frame)
            
            return {
                'success': True,
                'motion_detected': motion_detected,
                'motion_score': float(motion_score),
                'motion_cooldown': self.motion_cooldown,
                'buffer_size': len(self.frame_buffer),
                'is_stable': self.motion_cooldown == 0
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _detect_motion(self, current_frame):
        """
        Detect motion between consecutive frames.
        Returns: (motion_detected, motion_score)
        """
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return False, 0.0
        
        # Use ROI if available, else center of frame
        if self.forehead_roi is not None:
            fx, fy, fw, fh = self.forehead_roi
            curr_roi = current_frame[fy:fy+fh, fx:fx+fw]
            prev_roi = self.prev_frame[fy:fy+fh, fx:fx+fw]
        else:
            # Use center 50% of frame
            h, w = current_frame.shape[:2]
            cx, cy = w // 4, h // 4
            curr_roi = current_frame[cy:cy+h//2, cx:cx+w//2]
            prev_roi = self.prev_frame[cy:cy+h//2, cx:cx+w//2]
        
        # Calculate mean absolute difference
        diff = np.mean(np.abs(curr_roi.astype(float) - prev_roi.astype(float)))
        
        # Temporal filtering - require sustained motion
        self.motion_history.append(diff)
        avg_motion = np.mean(self.motion_history) if self.motion_history else diff
        
        motion_detected = avg_motion > self.motion_threshold
        
        self.prev_frame = current_frame.copy()
        
        return motion_detected, avg_motion
    
    def _detect_skin_tone(self, frame):
        """
        Detect skin tone category for population-based calibration.
        Uses HSV V-channel (Value/Brightness).
        """
        try:
            # Use center region of frame (likely face)
            h, w = frame.shape[:2]
            cy, cx = h // 2, w // 2
            roi = frame[cy-h//4:cy+h//4, cx-w//4:cx+w//4]
            
            # Convert to HSV
            hsv = self._rgb_to_hsv(roi)
            v_channel = hsv[:, :, 2].astype(float) / 255.0
            
            # Mean value
            mean_v = np.mean(v_channel)
            self.skin_tone_value = float(mean_v)
            
            # Categorize
            if mean_v > 0.7:
                self.skin_tone_category = 'light'
            elif mean_v > 0.4:
                self.skin_tone_category = 'medium'
            else:
                self.skin_tone_category = 'dark'
                
        except Exception as e:
            print(f"Skin tone detection error: {e}")
            self.skin_tone_category = 'medium'  # Default
            self.skin_tone_value = 0.5
    
    def extract_ppg_signal(self, use_green_only=False):
        """Extract PPG signal with POS algorithm."""
        if len(self.frame_buffer) < self.fps:
            return None
        
        try:
            frames = np.array(list(self.frame_buffer))
            
            # Detect face in middle frame
            mid_idx = len(frames) // 2
            face_found, forehead_roi = self._detect_face_and_forehead(frames[mid_idx])
            
            if not face_found:
                return None
            
            # Stabilize ROI (avoid jumping)
            if self.forehead_roi is not None:
                fx, fy, fw, fh = self.forehead_roi
                new_fx, new_fy, new_fw, new_fh = forehead_roi
                
                # Check if new ROI is close to previous
                if (abs(new_fx - fx) < 20 and abs(new_fy - fy) < 20 and
                    abs(new_fw - fw) < 20 and abs(new_fh - fh) < 20):
                    self.roi_stable_count += 1
                else:
                    self.roi_stable_count = 0
                    self.forehead_roi = forehead_roi
            else:
                self.forehead_roi = forehead_roi
                self.roi_stable_count = 1
            
            # Only proceed if ROI is stable
            if self.roi_stable_count < 3:
                return None
            
            fx, fy, fw, fh = self.forehead_roi
            
            # Extract RGB means
            rgb_means = []
            for frame in frames:
                roi = frame[fy:fy+fh, fx:fx+fw]
                mean_rgb = np.mean(roi, axis=(0, 1))
                rgb_means.append(mean_rgb)
            
            rgb_means = np.array(rgb_means)
            
            # Extract PPG
            if use_green_only:
                ppg_signal = rgb_means[:, 1]
                ppg_signal = self._detrend_signal(ppg_signal)
                ppg_signal = self._bandpass_filter(ppg_signal)
            else:
                ppg_signal = self._pos_algorithm(rgb_means)
            
            # Normalize
            ppg_signal = (ppg_signal - np.mean(ppg_signal)) / (np.std(ppg_signal) + 1e-10)
            
            # Quality check
            quality = self._calculate_signal_quality(ppg_signal)
            
            return {
                'signal': ppg_signal,
                'quality': quality,
                'rgb_means': rgb_means,
                'forehead_roi': self.forehead_roi,
                'roi_stable': self.roi_stable_count >= 3
            }
            
        except Exception as e:
            print(f"PPG extraction error: {e}")
            return None
    
    def calculate_vitals(self):
        """Calculate all vital signs with calibration."""
        ppg_data = self.extract_ppg_signal(use_green_only=False)
        
        if ppg_data is None:
            return None
        
        if ppg_data['quality'] < self.min_signal_quality:
            return {
                'success': False,
                'error': 'Poor signal quality',
                'quality': float(ppg_data['quality']),
                'buffer_size': len(self.frame_buffer)
            }
        
        ppg_signal = ppg_data['signal']
        
        # Calculate HR
        hr_data = self._calculate_heart_rate(ppg_signal)
        if hr_data is None:
            return None
        
        # Apply skin tone calibration to HR
        if self.skin_tone_category:
            offsets = self.calibration_offsets[self.skin_tone_category]
            hr_data['bpm'] = round(hr_data['bpm'] * offsets['hr_gain'], 1)
            hr_data['calibration_applied'] = self.skin_tone_category
        
        # Calculate SpO2
        spo2_data = self._calculate_spo2(ppg_data['rgb_means'])
        if spo2_data and self.skin_tone_category:
            offsets = self.calibration_offsets[self.skin_tone_category]
            spo2_data['spo2'] = round(spo2_data['spo2'] + offsets['spo2'], 1)
            spo2_data['calibration_applied'] = self.skin_tone_category
        
        # Calculate respiration
        resp_data = self._calculate_respiration(ppg_signal)
        
        self.measurement_count += 1
        
        result = {
            'success': True,
            'session_id': self.session_id,
            'measurement_id': self.measurement_count,
            'timestamp': int(datetime.now().timestamp() * 1000),
            'heart_rate': hr_data,
            'spo2': spo2_data,
            'respiration': resp_data,
            'signal_quality': round(ppg_data['quality'], 2),
            'skin_tone': {
                'category': self.skin_tone_category,
                'value': self.skin_tone_value
            },
            'motion_state': {
                'in_cooldown': self.motion_cooldown > 0,
                'cooldown_frames': self.motion_cooldown
            }
        }
        
        return result
    
    def _calculate_heart_rate(self, ppg_signal):
        """Calculate HR with peak detection."""
        try:
            min_peak_distance = int(self.fps * 0.4)
            peaks, properties = signal.find_peaks(
                ppg_signal,
                distance=min_peak_distance,
                prominence=0.2,
                width=(1, None)
            )
            
            if len(peaks) < 3:
                return None
            
            # Calculate IBIs
            ibis = np.diff(peaks) / self.fps
            valid_ibis = ibis[(ibis > 0.4) & (ibis < 1.5)]
            
            if len(valid_ibis) < 2:
                return None
            
            mean_ibi = np.mean(valid_ibis)
            bpm = 60.0 / mean_ibi
            sdnn = np.std(valid_ibis * 1000)
            
            peak_consistency = 1.0 - (np.std(valid_ibis) / np.mean(valid_ibis))
            confidence = peak_consistency * 100
            
            return {
                'bpm': round(bpm, 1),
                'confidence': round(confidence, 1),
                'ibi_ms': round(mean_ibi * 1000, 1),
                'hrv_sdnn': round(sdnn, 1),
                'num_beats': len(peaks)
            }
            
        except Exception as e:
            print(f"HR calculation error: {e}")
            return None
    
    def _calculate_spo2(self, rgb_means):
        """Calculate SpO2 with calibration."""
        try:
            red = rgb_means[:, 0]
            green = rgb_means[:, 1]
            
            red_ac = np.std(red)
            red_dc = np.mean(red)
            green_ac = np.std(green)
            green_dc = np.mean(green)
            
            if red_dc < 1 or green_dc < 1 or green_ac < 0.1:
                return None
            
            ratio = (red_ac / red_dc) / (green_ac / green_dc)
            spo2 = 110 - 25 * ratio
            spo2 = max(90, min(spo2, 100))
            
            confidence = min(100, max(0, 100 - abs(ratio - 1.0) * 50))
            
            return {
                'spo2': round(spo2, 1),
                'confidence': round(confidence, 1),
                'ratio': round(ratio, 3),
                'raw_value': True
            }
            
        except Exception as e:
            print(f"SpO2 calculation error: {e}")
            return None
    
    def _calculate_respiration(self, ppg_signal):
        """Calculate respiration rate."""
        try:
            analytic_signal = signal.hilbert(ppg_signal)
            amplitude_envelope = np.abs(analytic_signal)
            
            sos = signal.butter(4, [0.1, 0.5], btype='band', fs=self.fps, output='sos')
            resp_signal = signal.sosfiltfilt(sos, amplitude_envelope)
            
            fft_vals = np.abs(fft(resp_signal))
            freqs = fftfreq(len(resp_signal), 1/self.fps)
            
            valid_idx = (freqs >= 0.1) & (freqs <= 0.5)
            if not np.any(valid_idx):
                return None
            
            peak_idx = np.argmax(fft_vals[valid_idx])
            resp_freq = freqs[valid_idx][peak_idx]
            rpm = resp_freq * 60
            
            peak_power = fft_vals[valid_idx][peak_idx]
            total_power = np.sum(fft_vals[valid_idx])
            confidence = (peak_power / total_power) * 100
            
            return {
                'rpm': round(rpm, 1),
                'confidence': round(confidence, 1)
            }
            
        except Exception as e:
            print(f"Respiration calculation error: {e}")
            return None
    
    # Helper methods (same as before)
    def _rgb_to_hsv(self, rgb):
        """Convert RGB to HSV."""
        rgb = rgb.astype(np.float32) / 255.0
        maxc = np.max(rgb, axis=2)
        minc = np.min(rgb, axis=2)
        delta = maxc - minc
        
        h = np.zeros_like(maxc)
        mask = delta != 0
        
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        h = np.where((maxc == r) & mask, (60 * ((g - b) / delta) + 360) % 360, h)
        h = np.where((maxc == g) & mask, (60 * ((b - r) / delta) + 120), h)
        h = np.where((maxc == b) & mask, (60 * ((r - g) / delta) + 240), h)
        
        s = np.where(maxc != 0, delta / maxc, 0)
        v = maxc
        
        hsv = np.stack([h / 2, s * 255, v * 255], axis=2).astype(np.uint8)
        return hsv
    
    def _detect_face_and_forehead(self, frame):
        """Skin-based face detection."""
        try:
            hsv = self._rgb_to_hsv(frame)
            
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 150, 255], dtype=np.uint8)
            
            skin_mask = ((hsv[:, :, 0] >= lower_skin[0]) & (hsv[:, :, 0] <= upper_skin[0]) &
                        (hsv[:, :, 1] >= lower_skin[1]) & (hsv[:, :, 1] <= upper_skin[1]) &
                        (hsv[:, :, 2] >= lower_skin[2]) & (hsv[:, :, 2] <= upper_skin[2]))
            
            from scipy import ndimage
            labeled, num_features = ndimage.label(skin_mask)
            
            if num_features == 0:
                return False, None
            
            sizes = ndimage.sum(skin_mask, labeled, range(1, num_features + 1))
            largest_idx = np.argmax(sizes) + 1
            
            coords = np.argwhere(labeled == largest_idx)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            face_rect = (x_min, y_min, x_max - x_min, y_max - y_min)
            fx, fy, fw, fh = face_rect
            
            forehead_w = int(fw * 0.6)
            forehead_h = int(fh * 0.15)
            forehead_x = fx + (fw - forehead_w) // 2
            forehead_y = fy + int(fh * 0.1)
            
            height, width = frame.shape[:2]
            forehead_x = max(0, min(forehead_x, width - forehead_w))
            forehead_y = max(0, min(forehead_y, height - forehead_h))
            forehead_w = min(forehead_w, width - forehead_x)
            forehead_h = min(forehead_h, height - forehead_y)
            
            return True, (forehead_x, forehead_y, forehead_w, forehead_h)
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return False, None
    
    def _pos_algorithm(self, rgb):
        """POS algorithm implementation."""
        window_size = self.fps * 2
        stride = self.fps // 2
        
        ppg_segments = []
        
        for start in range(0, len(rgb) - window_size, stride):
            end = start + window_size
            window = rgb[start:end]
            
            mean_rgb = np.mean(window, axis=0)
            std_rgb = np.std(window, axis=0)
            std_rgb = np.where(std_rgb < 1e-10, 1, std_rgb)
            
            normalized = (window - mean_rgb) / std_rgb
            
            skin_tone = np.array([1, 0.5, 0.25])
            skin_tone = skin_tone / np.linalg.norm(skin_tone)
            
            projection = normalized - np.outer(np.dot(normalized, skin_tone), skin_tone)
            
            x = projection[:, 1]
            y = projection[:, 2]
            
            x_filtered = self._bandpass_filter(x)
            y_filtered = self._bandpass_filter(y)
            
            alpha = np.std(x_filtered) / (np.std(y_filtered) + 1e-10)
            segment_ppg = x_filtered - alpha * y_filtered
            
            ppg_segments.append(segment_ppg)
        
        if ppg_segments:
            full_ppg = np.concatenate(ppg_segments)
            return self._bandpass_filter(full_ppg)
        
        return np.zeros(len(rgb))
    
    def _bandpass_filter(self, data):
        """Apply bandpass filter."""
        nyquist = 0.5 * self.fps
        low = max(0.001, min(self.lowcut / nyquist, 0.99))
        high = max(low + 0.001, min(self.highcut / nyquist, 0.999))
        
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        return signal.sosfiltfilt(sos, data)
    
    def _detrend_signal(self, signal_data):
        """Remove trend."""
        return signal.detrend(signal_data)
    
    def _calculate_signal_quality(self, ppg_signal):
        """Calculate quality index."""
        try:
            fft_vals = np.abs(fft(ppg_signal))
            freqs = fftfreq(len(ppg_signal), 1/self.fps)
            
            hr_band = (freqs >= self.lowcut) & (freqs <= self.highcut)
            hr_power = np.sum(fft_vals[hr_band] ** 2)
            total_power = np.sum(fft_vals ** 2)
            
            snr = 10 * np.log10((hr_power + 1e-10) / (total_power - hr_power + 1e-10))
            snr_score = max(0, min(1, (snr + 10) / 30))
            
            if len(ppg_signal) > self.fps:
                autocorr = np.correlate(ppg_signal, ppg_signal, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]
                
                peaks, _ = signal.find_peaks(autocorr[1:], height=0.3, distance=self.fps//2)
                periodicity_score = peaks[0] / len(autocorr) if len(peaks) > 0 else 0
            else:
                periodicity_score = 0.5
            
            quality = 0.7 * snr_score + 0.3 * periodicity_score
            return quality
            
        except Exception as e:
            print(f"Quality calculation error: {e}")
            return 0.0
    
    def clear_buffer(self):
        """Clear all buffers."""
        self.frame_buffer.clear()
        self.timestamps.clear()
        self.prev_frame = None
        self.motion_history.clear()
        self.motion_cooldown = 0
        self.consecutive_motion_frames = 0
        self.forehead_roi = None
        self.roi_stable_count = 0
        self.skin_tone_category = None
        self.skin_tone_value = None


# ==================== Android Interface ====================

_processor = None

def get_processor():
    """Get singleton processor."""
    global _processor
    if _processor is None:
        _processor = VitalSignsProcessor()
    return _processor

def initialize(fps=30, buffer_seconds=5):
    """Initialize processor."""
    global _processor
    _processor = VitalSignsProcessor(fps=fps, buffer_seconds=buffer_seconds)
    return True

def start_session():
    """Start new session."""
    processor = get_processor()
    session_id = processor.start_session()
    return session_id

def end_session():
    """End session and return summary."""
    processor = get_processor()
    summary = processor.end_session()
    return json.dumps(summary)

def add_frame(frame_bytes, width, height, timestamp=None):
    """Add frame from Android."""
    processor = get_processor()
    result = processor.add_frame(frame_bytes, width, height, timestamp)
    return json.dumps(result)

def get_vitals():
    """Get vital signs."""
    try:
        processor = get_processor()
        result = processor.calculate_vitals()
        
        if result is None:
            return json.dumps({
                'success': False,
                'error': 'No valid data available',
                'buffer_size': len(processor.frame_buffer)
            })
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            'success': False,
            'error': str(e)
        })

def clear():
    """Clear buffers."""
    processor = get_processor()
    processor.clear_buffer()
    return json.dumps({'success': True})

def get_session_info():
    """Get current session info."""
    processor = get_processor()
    return json.dumps({
        'session_id': processor.session_id,
        'measurement_count': processor.measurement_count,
        'buffer_size': len(processor.frame_buffer),
        'skin_tone': {
            'category': processor.skin_tone_category,
            'value': processor.skin_tone_value
        },
        'motion_cooldown': processor.motion_cooldown
    })

def set_motion_threshold(threshold):
    """Adjust motion threshold."""
    processor = get_processor()
    processor.motion_threshold = float(threshold)
    return json.dumps({'success': True, 'threshold': threshold})
