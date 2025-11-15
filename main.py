# This is the full content of main.py (Version 3)

import os
import data_utils          # Your file for loading ECGs
import label_engineering   # Your teammate's file for processing labels
import numpy as np

# --- 1. Define File Paths ---
DATA_DIR = 'data'
CSV_PATH = os.path.join(DATA_DIR, 'exams.csv')
HDF5_PATH = os.path.join(DATA_DIR, 'exams_part0.hdf5')

# --- 2. Load Raw Data ---
# Load all ~233k labels from exams.csv
df_survival_all = data_utils.load_raw_labels(CSV_PATH)

# Load all ~20k available ECG IDs from exams_part0.hdf5
available_ecg_ids = data_utils.get_available_ids_from_hdf5(HDF5_PATH)
print(f"Found {len(available_ecg_ids)} available ECGs in {os.path.basename(HDF5_PATH)}.")

# --- 3. Find Common Data (The Intersection) ---
# Filter the DataFrame to get only the rows for which we have an ECG
df_final_to_process = df_survival_all[
    df_survival_all['exam_id'].isin(available_ecg_ids)
]
final_exam_ids_to_load = df_final_to_process['exam_id'].values

if len(final_exam_ids_to_load) == 0:
    print("FATAL ERROR: No exam_ids in common between CSV and HDF5 file.")
    exit()

print(f"Found {len(final_exam_ids_to_load)} exams that have BOTH a label AND an ECG file.")

# --- 4. Process Labels (Call Teammate's Code) ---
# We pass the final, filtered DataFrame to your teammate's code.
# This is Phase 3.
y_true, y_mask, exam_ids_from_labels = label_engineering.create_survival_labels(
    exams_df=df_final_to_process,
    n_intervals=120,          # We can configure this here
    max_time_years=10.0       # Or here
)

# --- 5. Process ECGs (Call Your Code) ---
# We pass the final, ordered list of IDs to your data loader.
# This is Phase 2.
X_data = data_utils.get_processed_ecgs_from_list(
    HDF5_PATH, 
    exam_ids_from_labels  # Use the ID list from the label encoder to guarantee order
)

# --- 6. Final Sanity Check ---
print("\n--- PROJECT SETUP SUCCESSFUL! ---")
print(f"Input X (ECGs) shape:  {X_data.shape}")
print(f"Output Y (Labels) shape: {y_true.shape}")
print(f"Mask (Masks) shape:  {y_mask.shape}")

# Final check that our data matches our labels perfectly
assert X_data.shape[0] == y_true.shape[0]

print("\nReady for Phase 4: Model Training.")