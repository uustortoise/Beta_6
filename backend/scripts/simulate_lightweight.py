import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, matthews_corrcoef

# 1. Simulate Imbalanced Data (98% Empty, 2% Activity) - Severe Imbalance
def generate_data(n_samples=10000, imbalance_ratio=0.98):
    # Class 0: Unoccupied (Easy, Frequent)
    # Class 1: Activity (Hard, Rare)
    
    n_class0 = int(n_samples * imbalance_ratio)
    n_class1 = n_samples - n_class0
    
    # Features: Slightly overlapping Gaussians
    # Class 0 centered at 0, Class 1 centered at 2
    X0 = np.random.normal(loc=0.0, scale=1.0, size=(n_class0, 10))
    X1 = np.random.normal(loc=2.0, scale=1.5, size=(n_class1, 10)) 
    
    X = np.concatenate([X0, X1])
    y = np.concatenate([np.zeros(n_class0), np.ones(n_class1)])
    
    # Shuffle
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    return X[indices], y[indices]

def simulate():
    print("Generating Synthetic Data (98% Imbalance)...")
    X, y = generate_data()
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print("\n--- Failure Mode 1: The 'Lazy' Model (Always Unoccupied) ---")
    y_pred_lazy = np.zeros_like(y_test) # Predicts 0 (Unoccupied) always
    # Note: sklearn might warn or error on 0 division
    print(classification_report(y_test, y_pred_lazy, target_names=['Unoccupied', 'Activity'], zero_division=0))
    mcc_lazy = matthews_corrcoef(y_test, y_pred_lazy)
    print(f"MCC: {mcc_lazy:.3f} (Correctly penalizes laziness)")
    
    print("\n--- Failure Mode 2: The 'Ghost' Model (Always Activity) ---")
    y_pred_ghost = np.ones_like(y_test) # Predicts 1 (Activity) always
    print(classification_report(y_test, y_pred_ghost, target_names=['Unoccupied', 'Activity'], zero_division=0))
    mcc_ghost = matthews_corrcoef(y_test, y_pred_ghost)
    print(f"MCC: {mcc_ghost:.3f} (Correctly penalizes hallucinations)")
    
    print("\n--- Baseline: Standard Training (No Weights) ---")
    clf_base = LogisticRegression(solver='lbfgs')
    clf_base.fit(X_train, y_train)
    y_pred_base = clf_base.predict(X_test)
    
    print(classification_report(y_test, y_pred_base, target_names=['Unoccupied', 'Activity'], zero_division=0))
    mcc_base = matthews_corrcoef(y_test, y_pred_base)
    print(f"MCC: {mcc_base:.3f}")
    
    print("\n--- Smart: Class Weighted (Balanced) ---")
    # Simulate "Focal Loss" effect by heavily weighting the minority class
    clf_smart = LogisticRegression(solver='lbfgs', class_weight='balanced')
    clf_smart.fit(X_train, y_train)
    y_pred_smart = clf_smart.predict(X_test)
    
    print(classification_report(y_test, y_pred_smart, target_names=['Unoccupied', 'Activity'], zero_division=0))
    mcc_smart = matthews_corrcoef(y_test, y_pred_smart)
    print(f"MCC: {mcc_smart:.3f}")
    
    print(f"\nGain in MCC: +{mcc_smart - mcc_base:.3f}")

if __name__ == "__main__":
    simulate()
