"""
Example Usage of Phase 3: Label Engineering

This script demonstrates how to use the label engineering module
with the CODE-15% dataset for your deep learning survival model.

Usage:
    python example_usage.py /path/to/exams.csv
"""

import sys
import pandas as pd
import numpy as np
from label_engineering import create_survival_labels, SurvivalLabelEncoder


def example_basic_usage():
    """Example 1: Basic usage with synthetic data."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Usage with Synthetic Data")
    print("=" * 70)

    # Create synthetic patient data
    synthetic_df = pd.DataFrame({
        'exam_id': [1001, 1002, 1003, 1004, 1005],
        'timey': [365.25, 730.5, 1826.25, 2000.0, 3652.5],  # 1y, 2y, 5y, 5.5y, 10y
        'death': [False, True, False, True, False]
    })

    print("\nInput data:")
    print(synthetic_df)

    # Create survival labels
    y_true, y_mask, exam_ids = create_survival_labels(
        synthetic_df,
        n_intervals=120,
        max_time_years=10.0
    )

    print(f"\nOutput:")
    print(f"  - y_true shape: {y_true.shape}")
    print(f"  - y_mask shape: {y_mask.shape}")
    print(f"  - exam_ids shape: {exam_ids.shape}")

    # Examine one patient
    print(f"\n--- Patient {exam_ids[1]} (died at 2 years) ---")
    patient_idx = 1
    interval_idx = int(np.where((y_mask[patient_idx] == 1) & (y_true[patient_idx] == 0))[0][0])
    masked_intervals = int(np.sum(y_mask[patient_idx]))

    print(f"  Masked intervals: {masked_intervals}")
    print(f"  Death occurred at interval: {interval_idx}")
    print(f"  First 30 intervals (mask): {y_mask[patient_idx][:30].astype(int)}")
    print(f"  First 30 intervals (true): {y_true[patient_idx][:30].astype(int)}")


def example_with_code15_dataset(exams_csv_path):
    """Example 2: Usage with actual CODE-15% dataset."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Usage with CODE-15% Dataset")
    print("=" * 70)

    # Load the exams.csv file
    print(f"\nLoading: {exams_csv_path}")
    exams_df = pd.read_csv(exams_csv_path)

    print(f"Dataset shape: {exams_df.shape}")
    print(f"Columns: {list(exams_df.columns)}")

    # Show sample
    print("\nFirst 5 rows:")
    print(exams_df[['exam_id', 'timey', 'death']].head())

    # Create survival labels
    print("\nCreating survival labels...")
    y_true, y_mask, exam_ids = create_survival_labels(
        exams_df,
        n_intervals=120,
        max_time_years=10.0,
        validate=True
    )

    print(f"\nSuccessfully created labels for {len(exam_ids)} patients!")

    # Statistics
    n_events = np.sum(np.any((y_mask == 1) & (y_true == 0), axis=1))
    n_censored = len(exam_ids) - n_events

    print(f"\nLabel Statistics:")
    print(f"  - Total patients: {len(exam_ids)}")
    print(f"  - Events (deaths): {n_events}")
    print(f"  - Censored: {n_censored}")
    print(f"  - Average masked intervals: {np.sum(y_mask, axis=1).mean():.1f}")

    return y_true, y_mask, exam_ids


def example_train_test_split(exams_df):
    """Example 3: Creating train/val/test splits."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Train/Val/Test Split")
    print("=" * 70)

    # Important: Split by patient_id, not exam_id!
    print("\nSplitting by patient_id to prevent data leakage...")

    unique_patients = exams_df['patient_id'].unique() if 'patient_id' in exams_df.columns else exams_df['exam_id'].unique()
    n_patients = len(unique_patients)

    # 70% train, 15% val, 15% test
    np.random.seed(42)
    shuffled_patients = np.random.permutation(unique_patients)

    train_size = int(0.70 * n_patients)
    val_size = int(0.15 * n_patients)

    train_patients = shuffled_patients[:train_size]
    val_patients = shuffled_patients[train_size:train_size + val_size]
    test_patients = shuffled_patients[train_size + val_size:]

    # Split DataFrames
    patient_col = 'patient_id' if 'patient_id' in exams_df.columns else 'exam_id'

    train_df = exams_df[exams_df[patient_col].isin(train_patients)]
    val_df = exams_df[exams_df[patient_col].isin(val_patients)]
    test_df = exams_df[exams_df[patient_col].isin(test_patients)]

    print(f"\nSplit sizes:")
    print(f"  - Train: {len(train_df)} exams ({len(train_patients)} patients)")
    print(f"  - Val:   {len(val_df)} exams ({len(val_patients)} patients)")
    print(f"  - Test:  {len(test_df)} exams ({len(test_patients)} patients)")

    # Create labels for each split
    print("\nCreating labels for each split...")

    y_train, mask_train, ids_train = create_survival_labels(train_df, validate=False)
    y_val, mask_val, ids_val = create_survival_labels(val_df, validate=False)
    y_test, mask_test, ids_test = create_survival_labels(test_df, validate=False)

    print("\nReady for training:")
    print(f"  - X_train shape: (should match) {y_train.shape}")
    print(f"  - y_train shape: {y_train.shape}")
    print(f"  - mask_train shape: {mask_train.shape}")

    return {
        'train': (y_train, mask_train, ids_train),
        'val': (y_val, mask_val, ids_val),
        'test': (y_test, mask_test, ids_test)
    }


def example_integration_with_model():
    """Example 4: How to use labels with Keras model (Phase 4)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Integration with Keras Model (Phase 4 Preview)")
    print("=" * 70)

    print("""
When you build your model in Phase 4, you'll use the labels like this:

```python
import tensorflow as tf
from tensorflow import keras

# Assume you have:
# - X_train: shape (n_samples, 4096, 8) - ECG signals
# - y_train: shape (n_samples, 120) - survival labels from Phase 3
# - mask_train: shape (n_samples, 120) - mask from Phase 3

# Build model
model = keras.Sequential([
    keras.layers.Input(shape=(4096, 8)),

    # Your 1D-CNN architecture here
    keras.layers.Conv1D(64, 7, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling1D(2),

    # ... more layers ...

    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(120, activation='sigmoid')  # 120 = n_intervals
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy'  # Applied per-interval
)

# Train with sample_weight to apply the mask!
history = model.fit(
    X_train,
    y_train,
    sample_weight=mask_train,  # This masks the loss!
    validation_data=(X_val, y_val, mask_val),
    epochs=50,
    batch_size=32
)

# Predict survival probabilities
y_pred = model.predict(X_test)  # shape: (n_test, 120)

# Calculate survival curves (cumulative product)
survival_curves = np.cumprod(y_pred, axis=1)

# Risk score at 10 years
risk_scores = 1 - survival_curves[:, -1]
```
""")


def main():
    """Main function."""
    print("=" * 70)
    print(" " * 15 + "PHASE 3: LABEL ENGINEERING")
    print(" " * 20 + "EXAMPLE USAGE")
    print("=" * 70)

    # Example 1: Basic usage
    example_basic_usage()

    # Example 2: CODE-15% dataset (if path provided)
    if len(sys.argv) > 1:
        exams_csv_path = sys.argv[1]
        try:
            y_true, y_mask, exam_ids = example_with_code15_dataset(exams_csv_path)

            # Load the DataFrame again for train/test split example
            exams_df = pd.read_csv(exams_csv_path)
            splits = example_train_test_split(exams_df)

        except FileNotFoundError:
            print(f"\n⚠ Warning: File not found: {exams_csv_path}")
            print("Skipping CODE-15% examples.")
        except Exception as e:
            print(f"\n⚠ Warning: Error loading dataset: {e}")
            print("Skipping CODE-15% examples.")
    else:
        print("\n" + "=" * 70)
        print("To run with your CODE-15% dataset:")
        print("  python example_usage.py /path/to/exams.csv")
        print("=" * 70)

    # Example 4: Integration preview
    example_integration_with_model()

    print("\n" + "=" * 70)
    print("Examples complete! Phase 3 is ready to use.")
    print("=" * 70)


if __name__ == "__main__":
    main()
