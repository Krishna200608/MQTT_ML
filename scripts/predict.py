import pandas as pd
import numpy as np
import joblib
import sys
import os 

# --- CONFIGURATION ---
# 1. Define the models you want to use (these must match the saved files)
# Let's use the models from Slice 0, Fold 0
BASE_OUTPUT_DIR = "Outputs"
EXPERIMENT_NAME = "Bidirectional_Results" # This is the folder you want to use
SLICE_NAME = "Slice_0"
PREFIX = "slice_0_k_0"
CLASSIFIER_NAME = "Random_Forest" # Use 'Random_Forest', 'Decision_Tree', 'SVM_RBF_Kernel' etc.

# 2. Define the feature mode
# 0 = Packet, 1 = Unidirectional, 2 = Bidirectional
MODE_TO_TEST = 2 

# 3. Define your new data file
NEW_DATA_FILE = "biflow_my_capture_WithWindowing.csv"
# NEW_DATA_FILE = "datasets/biflow_features/biflow_scan_A.csv"

# ---------------------

# --- NEW: Define the *exact* 28 features the model was trained on ---
# This fixes the "34 features vs 28 features" error.
TRAINING_COLUMNS_MODE_2 = [
    'prt_src', 'prt_dst', 'fwd_num_pkts', 'bwd_num_pkts', 'fwd_mean_iat', 
    'bwd_mean_iat', 'fwd_std_iat', 'bwd_std_iat', 'fwd_min_iat', 'bwd_min_iat', 
    'fwd_max_iat', 'bwd_max_iat', 'fwd_mean_pkt_len', 'bwd_mean_pkt_len', 
    'fwd_std_pkt_len', 'bwd_std_pkt_len', 'fwd_min_pkt_len', 'bwd_min_pkt_len', 
    'fwd_max_pkt_len', 'bwd_max_pkt_len', 'fwd_num_bytes', 'bwd_num_bytes', 
    'fwd_num_psh_flags', 'bwd_num_psh_flags', 'fwd_num_rst_flags', 
    'bwd_num_rst_flags', 'fwd_num_urg_flags', 'bwd_num_urg_flags'
]
# -----------------------------------------------------------------

def load_artifacts(base_dir, experiment_name, slice_name, prefix, mode):
    """Loads the saved model, scaler, and preprocessor."""
    
    slice_path = os.path.join(base_dir, experiment_name, slice_name)
    print(f"Loading artifacts from: {slice_path}")
    
    # 1. Load Model
    # Fixed model name cleaning to match save logic
    model_name_cleaned = CLASSIFIER_NAME.replace(' ', '_').replace('(', '').replace(')', '')
    model_filename = f"{prefix}_{model_name_cleaned}.joblib"
    model_path = os.path.join(slice_path, "Trained_Models", model_filename)
    
    if not os.path.isfile(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, None, None
    model = joblib.load(model_path)
    print(f"Successfully loaded model: {model_path}")

    # 2. Load Scaler
    scaler_filename = f"{prefix}_SCALER.joblib"
    # Scaler is saved in the slice root, not Trained_Models
    scaler_path = os.path.join(slice_path, scaler_filename) 
    if not os.path.isfile(scaler_path):
        print(f"Error: Scaler file not found at {scaler_path}")
        return None, None, None
    scaler = joblib.load(scaler_path)
    print(f"Successfully loaded scaler: {scaler_path}")
    
    # 3. Load Preprocessor (only for mode 0)
    preprocessor = None
    if mode == 0:
        preprocessor_filename = f"{prefix}_PREPROCESSOR.joblib"
        # Preprocessor is also in the slice root
        preprocessor_path = os.path.join(slice_path, preprocessor_filename) 
        if not os.path.isfile(preprocessor_path):
            print(f"Error: Preprocessor file not found at {preprocessor_path}")
            # This is OK if we are only running mode 1 or 2
            if mode == 0:
                print("Warning: Mode 0 requires a preprocessor, but none was found.")
        else:
            preprocessor = joblib.load(preprocessor_path)
            print(f"Successfully loaded preprocessor: {preprocessor_path}")
        
    return model, scaler, preprocessor

def prepare_data(data_path, mode, preprocessor, scaler):
    """Loads and transforms new data to be ready for prediction."""
    print(f"Loading and preparing new data from: {data_path}")
    if not os.path.isfile(data_path):
        print(f"Error: New data file not found at {data_path}")
        return None, None
        
    dataset = pd.read_csv(data_path, low_memory=False)
    
    # Keep original IPs for reporting, then drop
    original_ips = []
    if 'src_ip' in dataset.columns and 'dst_ip' in dataset.columns:
         original_ips = list(zip(dataset['src_ip'], dataset['dst_ip']))
    
    # --- START OF MODIFICATION ---
    if mode == 0:
        # (Handling for mode 0, which requires preprocessor)
        print("Mode 0 prediction not fully implemented in this example.")
        print("Please use Mode 1 or 2, as they are the recommended models.")
        return None, None # Placeholder
        
    else: # Mode 1 or 2
        # Explicitly select only the columns the model was trained on
        try:
            if mode == 2:
                x_df = dataset[TRAINING_COLUMNS_MODE_2].copy()
            else: # mode == 1 (Requires a different list of columns)
                print("Mode 1 columns not defined. Using Mode 2 logic.")
                # You would define TRAINING_COLUMNS_MODE_1 if you wanted to use it
                x_df = dataset[TRAINING_COLUMNS_MODE_2].copy()

        except KeyError as e:
            print(f"Error: The new data file is missing a required feature: {e}")
            print("This capture file cannot be processed as-is.")
            return None, None
        
        # Clean and fill any missing values just in case
        for col in x_df.columns:
            x_df[col] = pd.to_numeric(x_df[col], errors='coerce').fillna(-1)
        x = x_df.values
        
        # Scale all features
        x_final = scaler.transform(x)
    # --- END OF MODIFICATION ---
        
    return x_final, original_ips

def main():
    model, scaler, preprocessor = load_artifacts(BASE_OUTPUT_DIR, EXPERIMENT_NAME, SLICE_NAME, PREFIX, MODE_TO_TEST)
    
    if not model or not scaler:
        print("Failed to load necessary model artifacts. Exiting.")
        return

    x_new, ips = prepare_data(NEW_DATA_FILE, MODE_TO_TEST, preprocessor, scaler)
    
    if x_new is None:
        print("Failed to prepare data. Exiting.")
        return
        
    print(f"\nMaking predictions on {x_new.shape[0]} new samples...")
    
    # 4. Make Predictions
    predictions = model.predict(x_new)
    
    # 5. Show Results
    # We map the numeric labels back to human-readable names
    label_map = {
        0: "Benign (normal)",
        1: "Attack (scan_A)",
        2: "Attack (scan_sU)",
        3: "Attack (sparta)",
        4: "Attack (mqtt_bruteforce)"
    }
    
    results = pd.Series(predictions).map(label_map)
    print("\n--- Prediction Results ---")
    print(results.value_counts())
    
    # Example: Print details for the first 5 predictions
    print("\n--- First 5 Predictions ---")
    for i in range(5):
        if i < len(predictions):
            print(f"Sample {i+1}: Predicted as ==> {results[i]}")

if __name__ == "__main__":
    main()
