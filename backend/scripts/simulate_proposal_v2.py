import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, matthews_corrcoef, recall_score, precision_score
import warnings

# Suppress sklearn warnings for clean output
# warnings.filterwarnings("ignore")
print("Script starting...")

def generate_temporal_data(n_samples=10000, seed=42):
    """
    Generates data with temporal drift (Night vs Day).
    """
    np.random.seed(seed)
    
    # "Night" (0-50% of timeline): 99% Empty, 1% Activity (Ghost/Noise)
    n_night = int(n_samples * 0.5)
    n_night_act = int(n_night * 0.01)
    n_night_emp = n_night - n_night_act
    
    X_night_emp = np.random.normal(loc=0.0, scale=1.0, size=(n_night_emp, 10))
    X_night_act = np.random.normal(loc=0.8, scale=1.0, size=(n_night_act, 10)) # Harder overlap (0.0 vs 0.8)
    y_night = np.concatenate([np.zeros(n_night_emp), np.ones(n_night_act)])
    X_night = np.concatenate([X_night_emp, X_night_act])
    
    # "Day" (50-100% of timeline): 90% Empty, 10% Activity
    n_day = n_samples - n_night
    n_day_act = int(n_day * 0.10)
    n_day_emp = n_day - n_day_act
    
    X_day_emp = np.random.normal(loc=0.0, scale=1.0, size=(n_day_emp, 10))
    X_day_act = np.random.normal(loc=0.8, scale=1.0, size=(n_day_act, 10)) # Same activity signature
    y_day = np.concatenate([np.zeros(n_day_emp), np.ones(n_day_act)])
    X_day = np.concatenate([X_day_emp, X_day_act])
    
    # Combine sequentially (Temporal structure)
    X = np.concatenate([X_night, X_day])
    y = np.concatenate([y_night, y_day])
    
    return X, y

def apply_downsampling(X, y, drop_prob=0.90, seed=42):
    """
    Randomly drops 'Unoccupied' (class 0) samples.
    """
    np.random.seed(seed)
    mask = np.ones(len(y), dtype=bool)
    
    # Identify indices of Class 0
    zero_indices = np.where(y == 0)[0]
    
    # Select 90% to drop
    drop_indices = np.random.choice(zero_indices, size=int(len(zero_indices) * drop_prob), replace=False)
    
    mask[drop_indices] = False
    return X[mask], y[mask]

def run_simulation(n_runs=10):
    print(f"Running Simulation ({n_runs} runs)...")
    
    # Metrics Storage
    results = {
        "A_Current_Apparent": {"mcc": [], "rec_1": [], "prec_1": []},
        "A_Current_Actual":   {"mcc": [], "rec_1": [], "prec_1": []}, 
        "B_Proposed":         {"mcc": [], "rec_1": [], "prec_1": []}
    }
    
    for i in range(n_runs):
        seed = 42 + i
        X, y = generate_temporal_data(n_samples=5000, seed=seed)
        
        # Temporal Split (Last 20% is Val)
        split_idx = int(len(X) * 0.8)
        X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
        y_train_raw, y_val_raw = y[:split_idx], y[split_idx:]
        
        # --- Scenario A: The Bug (Downsample BEFORE Split) ---
        X_train_A, y_train_A = apply_downsampling(X_train_raw, y_train_raw, seed=seed)
        X_val_A_broken, y_val_A_broken = apply_downsampling(X_val_raw, y_val_raw, seed=seed) 
        
        model_A = LogisticRegression(solver='lbfgs') 
        model_A.fit(X_train_A, y_train_A)
        
        # Evaluate A (Apparent)
        y_pred_A_broken = model_A.predict(X_val_A_broken)
        results["A_Current_Apparent"]["mcc"].append(matthews_corrcoef(y_val_A_broken, y_pred_A_broken))
        results["A_Current_Apparent"]["rec_1"].append(recall_score(y_val_A_broken, y_pred_A_broken, pos_label=1, zero_division=0))
        results["A_Current_Apparent"]["prec_1"].append(precision_score(y_val_A_broken, y_pred_A_broken, pos_label=1, zero_division=0))

        # Evaluate A (Actual/Hidden) 
        y_pred_A_real = model_A.predict(X_val_raw)
        results["A_Current_Actual"]["mcc"].append(matthews_corrcoef(y_val_raw, y_pred_A_real))
        results["A_Current_Actual"]["rec_1"].append(recall_score(y_val_raw, y_pred_A_real, pos_label=1, zero_division=0))
        results["A_Current_Actual"]["prec_1"].append(precision_score(y_val_raw, y_pred_A_real, pos_label=1, zero_division=0))

        # --- Scenario B: The Fix (Downsample Train ONLY, Weighted) ---
        X_train_B, y_train_B = apply_downsampling(X_train_raw, y_train_raw, seed=seed)
        
        model_B = LogisticRegression(solver='lbfgs', class_weight='balanced')
        model_B.fit(X_train_B, y_train_B)
        
        # Evaluate B
        y_pred_B = model_B.predict(X_val_raw)
        results["B_Proposed"]["mcc"].append(matthews_corrcoef(y_val_raw, y_pred_B))
        results["B_Proposed"]["rec_1"].append(recall_score(y_val_raw, y_pred_B, pos_label=1, zero_division=0))
        results["B_Proposed"]["prec_1"].append(precision_score(y_val_raw, y_pred_B, pos_label=1, zero_division=0))

    # Summarize
    print("\n--- Simulation Results (Mean +/- Std Dev) ---")
    
    print("\n1. Current Pipeline (The Bug)")
    print("   (What we see on Dashboard vs Reality)")
    rec_app = np.mean(results["A_Current_Apparent"]["rec_1"])
    prec_app = np.mean(results["A_Current_Apparent"]["prec_1"])
    
    rec_act = np.mean(results["A_Current_Actual"]["rec_1"])
    prec_act = np.mean(results["A_Current_Actual"]["prec_1"])
    mcc_act = np.mean(results["A_Current_Actual"]["mcc"])
    
    print(f"   Apparent Recall:    {rec_app:.3f} (Inflated?)")
    print(f"   Actual Recall:      {rec_act:.3f}")
    print(f"   Actual Precision:   {prec_act:.3f}")
    print(f"   Actual MCC:         {mcc_act:.3f}")
    
    print("\n2. Proposed Pipeline (The Fix)")
    print("   (Downsample Train Only + Class Weights)")
    rec_prop = np.mean(results["B_Proposed"]["rec_1"])
    prec_prop = np.mean(results["B_Proposed"]["prec_1"])
    mcc_prop = np.mean(results["B_Proposed"]["mcc"])
    
    print(f"   Actual Recall:      {rec_prop:.3f}")
    print(f"   Actual Precision:   {prec_prop:.3f}")
    print(f"   Actual MCC:         {mcc_prop:.3f}")
    
    print("\n3. Estimated Gains")
    print(f"   Recall Gain:        +{(rec_prop - rec_act):.3f}")
    print(f"   Precision Gain:     +{(prec_prop - prec_act):.3f}")
    print(f"   MCC Gain:           +{(mcc_prop - mcc_act):.3f}")

if __name__ == "__main__":
    run_simulation()
