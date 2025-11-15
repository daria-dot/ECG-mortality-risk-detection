

# This is the full content of data_utils.py (Version 5)

import os
import h5py
import pandas as pd
import numpy as np

# --- Phase 2: Constants ---
LEADS_ALL = ['DI','DII','DIII','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
LEADS_REQUIRED = ['DI','DII','V1','V2','V3','V4','V5','V6']
LEAD_INDICES = [LEADS_ALL.index(lead) for lead in LEADS_REQUIRED]


def load_raw_labels(csv_path):
    """
    Phase 1: Loads the raw exams.csv file and returns the DataFrame.
    It also filters for rows that have mortality data.
    """
    print(f"Loading raw labels from: {csv_path}")
    df_labels = pd.read_csv(csv_path)
    
    # Filter for rows that have mortality data (death is not NaN)
    df_survival_all = df_labels[df_labels['death'].notna()].copy()
    print(f"Loaded {len(df_labels)} total exams, found {len(df_survival_all)} with mortality data.")
    
    return df_survival_all


def get_available_ids_from_hdf5(hdf5_path):
    """
    Helper function to quickly read all exam_ids from an HDF5 file.
    Returns a Python set for fast lookups.
    """
    print(f"Reading all available exam_ids from: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as hf:
        # Convert to a set for O(1) lookups, which is much faster
        return set(hf['exam_id'][:])


# --- THIS IS THE UPDATED FUNCTION ---

def get_processed_ecgs_from_list(hdf5_path, exam_ids_to_load):
    """
    Phase 2: Loads the HDF5 file and performs lead selection.
    
    This function is guaranteed that every ID in exam_ids_to_load
    is present in the HDF5 file.
    
    Args:
        hdf5_path: Path to the .hdf5 file (e.g., exams_part0.hdf5)
        exam_ids_to_load: A np.array of the exact exam_ids we want to load.
    
    Returns:
        X_data (np.array): The processed ECG data, shape (n_samples, 4096, 8)
    """
    print(f"Loading {len(exam_ids_to_load)} specific ECGs from: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as hf:
        all_exam_ids_in_file = hf['exam_id'][:]
        
        # Create a fast lookup map {exam_id -> hdf5_row_index}
        id_to_index_map = {id_val: index for index, id_val in enumerate(all_exam_ids_in_file)}
        
        # Get the HDF5 row numbers for the specific IDs we want
        # This list is in the order of exam_ids_to_load, so it is NOT sorted
        hdf5_indices_to_load = [id_to_index_map[id_val] for id_val in exam_ids_to_load]

        # --- START: h5py FIX (Version 2) ---
        
        # 1. Get the "sort key" (the indices that would sort the list)
        sort_key = np.argsort(hdf5_indices_to_load)
        
        # 2. Get the "unsort key" (the indices that will restore the original order)
        unsort_key = np.argsort(sort_key)
        
        # 3. Create the sorted list of HDF5 indices
        sorted_hdf5_indices = np.array(hdf5_indices_to_load)[sort_key]
        
        # 4. Load the data using the SORTED list.
        #    We can ONLY use ONE fancy index (sorted_hdf5_indices).
        #    So, we must load ALL 12 leads first.
        #    This is now a normal NumPy array in memory.
        X_data_sorted_12_lead = hf['tracings'][sorted_hdf5_indices, :, :]
        
        # 5. "Unsort" the data back to its original order (to align with labels).
        X_data_unsorted_12_lead = X_data_sorted_12_lead[unsort_key]

        # 6. Now that it's a normal NumPy array, we can select our 8 leads.
        X_data = X_data_unsorted_12_lead[:, :, LEAD_INDICES]
        
        # --- END: h5py FIX ---

    print(f"ECG processing complete. Final shape: {X_data.shape}")
    return X_data