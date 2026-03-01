# Beta 5.5 Machine Learning Audit Report

**Auditor:** Senior Engineer / Data Scientist  
**Date:** 2026-02-07  
**Scope:** ML Pipeline, Training, and Prediction Logic  
**Severity:** 🔴 Critical | 🟡 High | 🟢 Medium | ⚪ Low

---

## 🔴 CRITICAL ISSUES

### 1. No Class Imbalance Handling

**File:** `backend/ml/training.py:60-216`

**Issue:** Activity classification has severe class imbalance (e.g., 'inactive' vs 'shower'), but no class weights or resampling is applied.

```python
# Current code (training.py:160-166)
history = model.fit(X_seq, y_seq, 
          epochs=DEFAULT_EPOCHS, 
          batch_size=32,
          validation_split=DEFAULT_VALIDATION_SPLIT, 
          shuffle=True, 
          verbose=2,
          callbacks=callbacks)
# No class_weight parameter!
```

**Impact:**
- Model biased toward majority classes ('inactive', 'sleep')
- Poor recall on minority classes ('fall', 'shower')
- Misleading accuracy metrics (high accuracy due to majority class dominance)

**Fix:**
```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
classes = np.unique(y_seq)
class_weights = compute_class_weight('balanced', classes=classes, y=y_seq)
class_weight_dict = dict(enumerate(class_weights))

# Use in training
history = model.fit(X_seq, y_seq,
          class_weight=class_weight_dict,  # Add this
          epochs=DEFAULT_EPOCHS,
          ...)
```

**Priority:** 🔴 **CRITICAL** - Affects model fairness and minority class detection

---

### 2. Data Leakage in Temporal Sequences

**File:** `backend/ml/training.py:160-166`, `backend/ml/prediction.py:100-101`

**Issue:** `shuffle=True` in training shuffles temporal sequences, breaking temporal dependencies and causing data leakage between train/validation splits.

```python
# training.py:164
shuffle=True,  # <-- WRONG for time series!
```

**Impact:**
- Validation set contains future information (look-ahead bias)
- Inflated validation accuracy that doesn't generalize
- Model fails in production on truly unseen data

**Fix:**
```python
# For time series: DO NOT SHUFFLE
shuffle=False,

# Use temporal split instead of random split
# First 80% for training, last 20% for validation
train_size = int(len(X_seq) * 0.8)
X_train, X_val = X_seq[:train_size], X_seq[train_size:]
y_train, y_val = y_seq[:train_size], y_seq[train_size:]

model.fit(X_train, y_train, validation_data=(X_val, y_val), ...)
```

**Priority:** 🔴 **CRITICAL** - Invalidates all validation metrics

---

### 3. Single Metric Reporting (Accuracy Only)

**File:** `backend/ml/training.py:168-175`

**Issue:** Only tracking accuracy, ignoring precision, recall, F1, and per-class metrics.

```python
# training.py:169-174
final_acc = history.history['accuracy'][-1]
metrics = {
    'room': room_name,
    'accuracy': float(final_acc),  # Only accuracy!
    'epochs': DEFAULT_EPOCHS,
    'samples': len(X_seq)
}
```

**Impact:**
- Cannot detect poor minority class performance
- Misleading model quality assessment
- No visibility into class-specific errors

**Fix:**
```python
from sklearn.metrics import classification_report, confusion_matrix

# After training, compute full metrics
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

report = classification_report(y_val, y_pred_classes, output_dict=True)
metrics = {
    'room': room_name,
    'accuracy': float(report['accuracy']),
    'macro_precision': float(report['macro avg']['precision']),
    'macro_recall': float(report['macro avg']['recall']),
    'macro_f1': float(report['macro avg']['f1-score']),
    'per_class': {k: v for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']},
    'confusion_matrix': confusion_matrix(y_val, y_pred_classes).tolist()
}
```

**Priority:** 🔴 **CRITICAL** - Blind to model failures on critical classes like 'fall'

---

## 🟡 HIGH SEVERITY ISSUES

### 4. No Early Stopping

**File:** `backend/ml/training.py:160-166`

**Issue:** Fixed 5 epochs with no early stopping - leads to overfitting.

**Fix:**
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

callbacks = [early_stopping]  # Add to existing callbacks
```

---

### 5. Validation Split on Small Datasets

**File:** `backend/ml/training.py:163`

**Issue:** `validation_split=0.2` fails on small datasets (<100 samples).

```python
# Example: 50 samples with 0.2 split = only 10 validation samples
# With many classes, some classes may have 0 validation samples!
```

**Fix:**
```python
# Ensure minimum validation size
min_val_samples = max(int(len(X_seq) * 0.2), 10)  # At least 10 samples
if min_val_samples >= len(X_seq):
    # Too small for validation split - use all for training
    validation_split = 0.0
```

---

### 6. Silent Label Encoder Failures

**File:** `backend/ml/training.py:424-430`

```python
try:
    encoded_label = self.platform.label_encoders[room_name].transform([corr_label])[0]
except ValueError:
    continue  # <-- Silent failure!
```

**Issue:** Golden samples with unseen labels are silently skipped without logging.

**Fix:**
```python
except ValueError as e:
    logger.warning(f"Skipping correction: label '{corr_label}' not in encoder for {room_name}")
    continue
```

---

### 7. Padding Strategy Issues

**File:** `backend/ml/training.py:392-398`

```python
if len(window_df) > room_seq_length:
    window_df = window_df.tail(room_seq_length)  # Truncate from end
elif len(window_df) < room_seq_length:
    # Pad with first row
    pad_count = room_seq_length - len(window_df)
    pad_df = pd.concat([window_df.head(1)] * pad_count, ignore_index=True)
    window_df = pd.concat([pad_df, window_df], ignore_index=True)  # Pre-pad
```

**Issue:** 
- Truncating from end loses the target timestamp data
- Pre-padding with first row creates artificial leading data

**Impact:**
- Sequence doesn't actually contain the correction timestamp
- Model trained on padded artificial data

**Fix:**
```python
# For corrections, we need the timestamp AT THE END of the sequence
if len(window_df) > room_seq_length:
    # Keep the END (most recent) data including correction timestamp
    window_df = window_df.iloc[-room_seq_length:]
elif len(window_df) < room_seq_length:
    # Pad at BEGINNING (pre-pad) with first valid row
    pad_count = room_seq_length - len(window_df)
    pad_df = pd.concat([window_df.iloc[[0]]] * pad_count, ignore_index=True)
    window_df = pd.concat([pad_df, window_df], ignore_index=True)
```

---

## 🟢 MEDIUM SEVERITY ISSUES

### 8. No Learning Rate Scheduling

**File:** `backend/ml/training.py:142-145`

Model uses fixed learning rate. Add decay for better convergence.

**Fix:**
```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, ...)
```

---

### 9. Hardcoded Batch Size

**File:** `backend/ml/training.py:162`

```python
batch_size=32,  # Hardcoded
```

**Issue:** May be too large for small datasets, too small for large datasets.

**Fix:**
```python
# Adaptive batch size
batch_size = min(32, max(8, len(X_seq) // 10))  # At least 10 batches
```

---

### 10. No Model Checkpointing

**File:** `backend/ml/training.py:160-166`

Model saves only at end, not best during training.

**Fix:**
```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    f'{room_name}_best.keras',
    monitor='val_accuracy',
    save_best_only=True
)
```

---

## ⚪ LOW SEVERITY (Code Quality)

### 11. Duplicate Imports

**File:** `backend/ml/training.py:8-19`

```python
import pandas as pd  # Line 2 and 17
from datetime import datetime, timedelta  # Line 7 and 18
from typing import Dict, List, Tuple, Any  # Line 8 and 19
```

### 12. Missing Input Validation

**File:** `backend/ml/training.py:60`

No validation that `processed_df` has required columns before training.

### 13. Hardcoded Architecture Parameters

**File:** `backend/ml/training.py:129-139`

Transformer parameters hardcoded - no configurability.

---

## 📊 DATA QUALITY CONCERNS

### 14. No Class Distribution Logging

**File:** `backend/ml/training.py:85-90`

```python
# Before training, should log class distribution
unique, counts = np.unique(y_seq, return_counts=True)
logger.info(f"Class distribution: {dict(zip(unique, counts))}")
```

### 15. Sequence Creation Without Overlap Check

**File:** `backend/ml/training.py:89`

Sequences are created with stride=1, but no check for minimum samples per class.

---

## 🎯 RECOMMENDATIONS SUMMARY

### Immediate Actions (Before Production)

1. **Add class weights** to handle imbalance (Critical for fall detection)
2. **Fix shuffle=False** for time series (Critical for valid validation)
3. **Add comprehensive metrics** beyond accuracy (Critical for monitoring)
4. **Fix padding strategy** for golden samples (High)

### Testing Strategy

```python
# Test class distribution
y_test = np.array([0, 0, 0, 0, 1, 2])  # 4:1:1 ratio
class_weights = compute_class_weight('balanced', classes=np.unique(y_test), y=y_test)
assert class_weights[0] < class_weights[1], "Majority class should have lower weight"

# Test temporal split
X = np.arange(100)
train_size = 80
X_train, X_val = X[:train_size], X[train_size:]
assert X_train[-1] < X_val[0], "No overlap between train and validation"
```

### Model Monitoring Dashboard

Track these metrics per room:
- Accuracy (current)
- Macro F1-score (new)
- Per-class precision/recall (new)
- Class distribution in predictions vs training (new)
- Low confidence rate (existing)

---

## VERIFICATION CHECKLIST

- [ ] Class weights applied in training
- [ ] Shuffle=False for time series
- [ ] Comprehensive metrics logged
- [ ] Early stopping implemented
- [ ] Per-class performance visible
- [ ] Validation on temporal holdout
- [ ] Class distribution balanced or weighted

---

**Most Critical:** Issues #1, #2, #3 affect model validity and safety. Fix before production.
