
# This is the full content of model_utils.py (Version 7 - Cleaned)
# This file ONLY contains data splitting and evaluation helpers.
# All model architecture and loss functions are now in model_architecture.py.

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from typing import Tuple

# --- Phase 5: Data Splitting Function ---
def split_data_by_patient(
    df_with_patients: pd.DataFrame,
    X_data: np.ndarray,
    y_true: np.ndarray,
    y_mask: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15
) -> Tuple:
    """
    Splits data by PATIENT_ID to prevent data leakage.
    
    This function returns 'y' data that is stacked (y_true, y_mask)
    for use in the custom loss function (which expects a packed tensor).
    """
    print(f"\nSplitting {len(df_with_patients)} exams by patient_id...")
    
    # 1. Get unique patient IDs
    unique_patient_ids = df_with_patients['patient_id'].unique()
    print(f"Found {len(unique_patient_ids)} unique patients.")

    # 2. Split patient IDs
    train_val_pids, test_pids = train_test_split(
        unique_patient_ids, test_size=test_size, random_state=42
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_pids, val_pids = train_test_split(
        train_val_pids, test_size=relative_val_size, random_state=42
    )

    print(f"  - Train patients: {len(train_pids)}")
    print(f"  - Val patients:   {len(val_pids)}")
    print(f"  - Test patients:  {len(test_pids)}")

    # 3. Get the *row indices* from the original DataFrame for each set
    train_indices = df_with_patients[df_with_patients['patient_id'].isin(train_pids)].index.values
    val_indices = df_with_patients[df_with_patients['patient_id'].isin(val_pids)].index.values
    test_indices = df_with_patients[df_with_patients['patient_id'].isin(test_pids)].index.values
    
    # 4. Use these indices to slice our main data arrays
    X_train = X_data[train_indices]
    X_val = X_data[val_indices]
    X_test = X_data[test_indices]
    
    # --- Stack y_true and y_mask together ---
    # We pack them into a single tensor with 2 channels.
    # We must cast to float32 for TensorFlow
    y_train_packed = np.stack([y_true[train_indices], y_mask[train_indices]], axis=-1).astype(np.float32)
    y_val_packed = np.stack([y_true[val_indices], y_mask[val_indices]], axis=-1).astype(np.float32)
    y_test_packed = np.stack([y_true[test_indices], y_mask[test_indices]], axis=-1).astype(np.float32)
    
    df_test = df_with_patients.loc[test_indices]
    
    print("Data splitting complete.")
    return (X_train, y_train_packed,
            X_val, y_val_packed,
            X_test, y_test_packed,
            df_test)


# --- Phase 5: Evaluation Function ---
def calculate_c_index(model, X_test, df_test):
    """
    Evaluates the model using the Concordance Index (C-index).
    
    Args:
        model: The trained Keras model.
        X_test: The input ECGs for the test set.
        df_test: The test DataFrame containing true 'timey' and 'death' columns.
    """
    print("\nCalculating C-index on test set...")
    
    # 1. Get model predictions
    # This will run in batches by default, which is good
    y_pred = model.predict(X_test, batch_size=32)
    
    # 2. Calculate cumulative survival (S_j)
    # This is the product of survival probabilities up to each interval
    cumulative_survival = np.cumprod(y_pred, axis=1)
    
    # 3. Get survival probability at 10 years (the last interval)
    # This will be our "risk score"
    # A high score (e.g., 0.95) means low risk
    risk_score_survival_prob = cumulative_survival[:, -1]
    
    # 4. Get true values from the test DataFrame
    event_times = df_test['timey'].values
    event_observed = df_test['death'].values
    
    # 5. Calculate C-index
    # We pass the NEGATIVE of the risk score.
    # lifelines expects a "risk" score (higher = worse outcome)
    # Our score is a "survival" score (higher = better outcome)
    # So, we flip it by passing -risk_score_survival_prob.
    c_index = concordance_index(
        event_times, 
        -risk_score_survival_prob, 
        event_observed
    )
    
    print(f"--- C-Index (Test Set): {c_index:.4f} ---")
    return c_index

if __name__ == "__main__":
    # This code only runs if you execute `python3 model_utils.py` directly
    print("="*50)
    print("This is model_utils.py")
    print("It contains helper functions for splitting and evaluating data.")
    print("Run main.py to execute the full pipeline.")
    print("="*50)