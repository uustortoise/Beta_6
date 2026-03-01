import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt

# 1. Simulate Imbalanced Data (95% Empty, 5% Activity)
def generate_data(n_samples=10000, imbalance_ratio=0.95):
    # Class 0: Unoccupied (Easy, Frequent)
    # Class 1: Activity (Hard, Rare)
    
    n_class0 = int(n_samples * imbalance_ratio)
    n_class1 = n_samples - n_class0
    
    # Features: Class 0 is clustered around 0, Class 1 around 1, but with overlap (noise)
    X0 = np.random.normal(loc=0.0, scale=1.0, size=(n_class0, 10))
    X1 = np.random.normal(loc=1.5, scale=1.0, size=(n_class1, 10)) # Slight overlap
    
    X = np.concatenate([X0, X1])
    y = np.concatenate([np.zeros(n_class0), np.ones(n_class1)])
    
    # Shuffle
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    return X[indices], y[indices]

# 2. Focal Loss Function
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) -K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

# Simpler Keras Version
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Binary Cross Entropy parts
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Focal weights
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = self.alpha * tf.pow(1.0 - pt, self.gamma)
        
        return tf.reduce_mean(focal_weight * bce)

# 2. Model Builder
def build_model_weighted(loss_fn, class_weight=None):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model

# 4. Run Simulation
def simulate():
    print("Generating Synthetic Data (95% Imbalance)...")
    X, y = generate_data()
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Training Baseline (Cross-Entropy)...")
    model_ce = build_model_weighted('binary_crossentropy')
    model_ce.fit(X_train, y_train, epochs=20, verbose=0, batch_size=32)
    
    print(f"Training Weighted Model (Class Weights)...")
    # Calculate weights: Total / (2 * Count)
    n_0 = len(y_train) - y_train.sum()
    n_1 = y_train.sum()
    weight_0 = (len(y_train) / (2 * n_0))
    weight_1 = (len(y_train) / (2 * n_1))
    class_weights = {0: weight_0, 1: weight_1}
    print(f"Weights: {class_weights}")
    
    model_w = build_model_weighted('binary_crossentropy')
    model_w.fit(X_train, y_train, epochs=20, verbose=0, batch_size=32, class_weight=class_weights)
    
    # Evaluate
    print("\n--- Results ---")
    
    # Baseline
    y_prob_ce = model_ce.predict(X_test)
    y_pred_ce = (y_prob_ce > 0.5).astype(int)
    print("\n[Baseline] Cross Entropy (No Weights):")
    print(classification_report(y_test, y_pred_ce, target_names=['Unoccupied', 'Activity']))
    mcc_ce = matthews_corrcoef(y_test, y_pred_ce)
    print(f"MCC: {mcc_ce:.3f}")
    
    # Weighted
    y_prob_w = model_w.predict(X_test)
    y_pred_w = (y_prob_w > 0.5).astype(int)
    print("\n[Smart] Class Weighted:")
    print(classification_report(y_test, y_pred_w, target_names=['Unoccupied', 'Activity']))
    mcc_w = matthews_corrcoef(y_test, y_pred_w)
    print(f"MCC: {mcc_w:.3f}")
    
    print(f"\nGain in MCC: +{mcc_w - mcc_ce:.3f}")

if __name__ == "__main__":
    simulate()
