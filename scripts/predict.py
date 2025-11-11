#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust predict.py that auto-adjusts uniflow/biflow columns to match saved scaler.
"""

import pandas as pd
import numpy as np
import joblib
import os
import os
# Ensure results directory exists
os.makedirs("./captures/results", exist_ok=True)
import sys

# -------------------------
# CONFIG
# -------------------------
BASE_OUTPUT_DIR = "Outputs"
EXPERIMENT_NAME = "Unidirectional_Results"   # folder where mode-1 models were saved
SLICE_NAME = "Slice_0"
PREFIX = "slice_0_k_0"
CLASSIFIER_NAME = "Random_Forest"
MODE_TO_TEST = 1
NEW_DATA_FILE = "captures/csv/uniflow_benign_mqtt.csv"

# Candidate feature lists
TRAINING_COLUMNS_MODE_1 = [
    'ip_src', 'ip_dst', 'prt_src', 'prt_dst', 'proto',
    'num_pkts', 'mean_iat', 'std_iat', 'min_iat', 'max_iat',
    'mean_offset', 'mean_pkt_len', 'std_pkt_len',
    'min_pkt_len', 'max_pkt_len', 'num_bytes',
    'num_psh_flags', 'num_rst_flags', 'num_urg_flags'
]

TRAINING_COLUMNS_MODE_2 = [
    'prt_src', 'prt_dst', 'fwd_num_pkts', 'bwd_num_pkts', 'fwd_mean_iat',
    'bwd_mean_iat', 'fwd_std_iat', 'bwd_std_iat', 'fwd_min_iat', 'bwd_min_iat',
    'fwd_max_iat', 'bwd_max_iat', 'fwd_mean_pkt_len', 'bwd_mean_pkt_len',
    'fwd_std_pkt_len', 'bwd_std_pkt_len', 'fwd_min_pkt_len', 'bwd_min_pkt_len',
    'fwd_max_pkt_len', 'bwd_max_pkt_len', 'fwd_num_bytes', 'bwd_num_bytes',
    'fwd_num_psh_flags', 'bwd_num_psh_flags', 'fwd_num_rst_flags',
    'bwd_num_rst_flags', 'fwd_num_urg_flags', 'bwd_num_urg_flags'
]

# Columns we prefer to drop first if necessary (identifiers)
PREFERRED_DROP_ORDER_MODE1 = ['ip_src', 'ip_dst', 'proto', 'mean_offset']

# -------------------------
# Helpers
# -------------------------
def load_artifacts(base_dir, experiment_name, slice_name, prefix, classifier_name, mode):
    slice_path = os.path.join(base_dir, experiment_name, slice_name)
    print(f"[INFO] Loading artifacts from: {os.path.abspath(slice_path)}")

    model_filename = f"{prefix}_{classifier_name}.joblib"
    model_path = os.path.join(slice_path, "Trained_Models", model_filename)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    print(f"✅ Loaded model: {model_filename}")

    scaler_filename = f"{prefix}_SCALER.joblib"
    scaler_path = os.path.join(slice_path, scaler_filename)
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print(f"✅ Loaded scaler: {scaler_filename}")

    return model, scaler

def choose_columns_for_scaler(df_columns, training_candidates, scaler_expected):
    """
    Attempt to choose and order columns from df_columns that match scaler_expected count.
    Strategy:
      1. Take intersection in candidate order.
      2. If too many, drop preferred identifier columns in order until match.
      3. If still too many, take first N (warn).
      4. If too few, error and list missing columns.
    """
    # keep only those candidates present in the CSV, preserve candidate order
    present = [c for c in training_candidates if c in df_columns]

    if len(present) == scaler_expected:
        return present, []

    # If we have more than expected: drop preferred identifiers first
    if len(present) > scaler_expected:
        present_mod = present.copy()
        dropped = []
        for col in PREFERRED_DROP_ORDER_MODE1:
            if col in present_mod and len(present_mod) > scaler_expected:
                present_mod.remove(col)
                dropped.append(col)
        # if still too many, trim from the end (least safe, but predictable)
        if len(present_mod) > scaler_expected:
            extra = len(present_mod) - scaler_expected
            # drop last extra columns
            dropped += present_mod[-extra:]
            present_mod = present_mod[:-extra]
        return present_mod, dropped

    # If we have fewer than expected: list missing
    missing = [c for c in training_candidates if c not in df_columns]
    return None, missing

# -------------------------
# Main prepare & predict
# -------------------------
def prepare_data_and_align(file_path, mode, scaler):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"New data file not found: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"[INFO] Loaded new data: {len(df)} rows, {len(df.columns)} columns")

    # choose candidate list by mode
    candidates = TRAINING_COLUMNS_MODE_1 if mode == 1 else TRAINING_COLUMNS_MODE_2
    scaler_expected = getattr(scaler, "n_features_in_", None)
    if scaler_expected is None:
        # older scaler versions may not have attribute; attempt to infer from shape_ if model saved same time
        raise RuntimeError("Scaler does not expose n_features_in_. Cannot auto-align. Please provide correct TRAINING_COLUMNS.")
    print(f"[INFO] Scaler expects {scaler_expected} features.")

    chosen_cols, info = choose_columns_for_scaler(df.columns, candidates, scaler_expected)

    if chosen_cols is None:
        # too few columns present
        missing = info
        raise KeyError(f"Missing required training columns for this model. Missing columns (candidates not in CSV): {missing}\n"
                       f"CSV columns: {list(df.columns)}\n"
                       f"Training candidates: {candidates}")
    else:
        dropped_or_trimmed = info  # either dropped list or empty

    print(f"[INFO] Selected {len(chosen_cols)} features for scaling: {chosen_cols}")
    if dropped_or_trimmed:
        print(f"[WARN] The script removed/trimmed these columns to match scaler: {dropped_or_trimmed}")

    # Build the feature matrix preserving chosen_cols order
    x_df = df[chosen_cols].copy()
    # coerce numeric
    for col in x_df.columns:
        x_df[col] = pd.to_numeric(x_df[col], errors='coerce').fillna(-1)

    # sanity: shape matches scaler
    if x_df.shape[1] != scaler_expected:
        raise RuntimeError(f"After alignment, feature count {x_df.shape[1]} != scaler expects {scaler_expected}")

    x_scaled = scaler.transform(x_df.values)
    print("✅ Features aligned and scaled.")
    return x_scaled

def predict_and_report(model, x):
    preds = model.predict(x)
    label_map = {0: "Benign (normal)",
                 1: "Attack (scan_A)",
                 2: "Attack (scan_sU)",
                 3: "Attack (sparta)",
                 4: "Attack (mqtt_bruteforce)"}
    results = pd.Series(preds).map(label_map)
    print("\n--- Prediction Summary ---")
    print(results.value_counts())
    # save
    out = os.path.join("./captures/results", os.path.splitext(os.path.basename(NEW_DATA_FILE))[0] + "_predictions.csv")
    results.to_csv(out, index=False)
    print(f"\n✅ Predictions saved to: {out}")
    return results

def main():
    try:
        model, scaler = load_artifacts(BASE_OUTPUT_DIR, EXPERIMENT_NAME, SLICE_NAME, PREFIX, CLASSIFIER_NAME, MODE_TO_TEST)
        x = prepare_data_and_align(NEW_DATA_FILE, MODE_TO_TEST, scaler)
        predict_and_report(model, x)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
