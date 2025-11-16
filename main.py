
# This is the full content of main.py (Version 7 - Clean Structure)
# This file imports from all 4 of our utility scripts.

import os
import data_utils          # Your file for loading ECGs
import label_engineering   # Your teammate's file for processing labels
import model_utils         # Your file for splitting/evaluating
import model_architecture  # Your teammate's file for the model
import numpy as np
import tensorflow as tf

# --- Configuration ---
DATA_DIR = 'data'
CSV_PATH = os.path.join(DATA_DIR, 'exams.csv')
HDF5_PATH = os.path.join(DATA_DIR, 'exams_part0.hdf5')

# Model Hyperparameters
EPOCHS = 10       # Start with 10 for a test, 50 for a full run
BATCH_SIZE = 32   # As per the paper
LEARNING_RATE = 1e-4 # Tuned learning rate

def run_pipeline():
    # --- 1. Load Raw Data (Phases 1 & 3) ---
    df_survival_all = data_utils.load_raw_labels(CSV_PATH)
    available_ecg_ids = data_utils.get_available_ids_from_hdf5(HDF5_PATH)
    print(f"Found {len(available_ecg_ids)} available ECGs in {os.path.basename(HDF5_PATH)}.")

    # --- 2. Find Common Data (The Intersection) ---
    df_final_to_process = df_survival_all[
        df_survival_all['exam_id'].isin(available_ecg_ids)
    ]
    df_final_to_process = df_final_to_process.reset_index(drop=True)
    final_exam_ids_to_load = df_final_to_process['exam_id'].values

    if len(final_exam_ids_to_load) == 0:
        print("FATAL ERROR: No exam_ids in common between CSV and HDF5 file.")
        return

    print(f"Found {len(final_exam_ids_to_load)} exams that have BOTH a label AND an ECG file.")

    # --- 3. Process Labels (Call Teammate's Code) ---
    y_true, y_mask, exam_ids_from_labels = label_engineering.create_survival_labels(
        exams_df=df_final_to_process,
        n_intervals=model_architecture.N_INTERVALS, # Use constant from model_architecture
        verbose=True
    )

    # --- 4. Process ECGs (Call Your Code) ---
    X_data = data_utils.get_processed_ecgs_from_list(
        HDF5_PATH, 
        exam_ids_from_labels  # Use the ID list from the label encoder
    )

    print("\n--- Data Loading Successful ---")
    print(f"X_data shape: {X_data.shape}, y_true shape: {y_true.shape}")

    # --- 5. Split Data (Call model_utils) ---
    (X_train, y_train_packed,
     X_val, y_val_packed,
     X_test, y_test_packed,
     df_test) = model_utils.split_data_by_patient(
         df_final_to_process, X_data, y_true, y_mask
    )
    
    print(f"Train set: {X_train.shape[0]} exams")
    print(f"Val set:   {X_val.shape[0]} exams")
    print(f"Test set:  {X_test.shape[0]} exams")

    # --- 6. Build & Train Model (Call model_architecture) ---
    model = model_architecture.compile_ecg_survival_model(
        learning_rate=LEARNING_RATE
    )
    
    model.summary()

    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train,
        y_train_packed, # <-- We pass the PACKED array
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val_packed), # <-- Pass packed validation data
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]
    )

    # --- 7. Evaluate Model (Call model_utils) ---
    print("\n--- Training Complete ---")
    
    c_index = model_utils.calculate_c_index(model, X_test, df_test)

    # --- 8. Save the Model ---
    print("\n--- Saving trained model ---")
    model.save(os.path.join(DATA_DIR, 'ecg_survival_model.h5'))
    print(f"Model saved to: {os.path.join(DATA_DIR, 'ecg_survival_model.h5')}")

    print("\n--- FULL PIPELINE COMPLETE ---")

# This makes the script runnable from the command line
if __name__ == "__main__":
    
    try:
        import lifelines
    except ImportError:
        print("="*50)
        print("ERROR: `lifelines` library not found.")
        print("Please run: pip install lifelines")
        print("="*50)
        exit()
        
    run_pipeline()