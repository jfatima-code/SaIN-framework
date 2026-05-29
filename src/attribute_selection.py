# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 23:16:38 2026

@author: Javaria
"""

import os
import pandas as pd

# ================= USER INPUT =================
ROOT_DIR = r"...\outputs_final_last"

METHODS = {
    "laplacian": "selected_features_lap.xlsx",
    "mcfs": "selected_features_mcfs.xlsx",
    "spec": "selected_features_spec.xlsx",
    "pca": "selected_features_pca.xlsx",
    "udfs": "selected_features_udfsBB.xlsx",
}

OUTPUT_FILE = os.path.join(ROOT_DIR, "All_Networks_Features.xlsx")
# ==============================================


def get_feature_names(file_path):
    """Read only column headers (feature names) from Excel."""
    df = pd.read_excel(file_path)
    return list(df.columns)


with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    # Loop over each NETWORK
    for network in os.listdir(ROOT_DIR):
        network_path = os.path.join(ROOT_DIR, network)

        if not os.path.isdir(network_path):
            continue

        print(f"📡 Processing network: {network}")

        method_features = {}

        # Loop over METHODS
        for method, fname in METHODS.items():
            file_path = os.path.join(network_path, method, fname)

            if not os.path.exists(file_path):
                print(f"⚠ Missing file: {file_path}")
                continue

            method_features[method] = get_feature_names(file_path)

        if not method_features:
            print(f"❌ No valid method files found for {network}")
            continue

        # -------- Build sheet --------
        max_len = max(len(v) for v in method_features.values())
        sheet_df = pd.DataFrame()

        for method, features in method_features.items():
            sheet_df[method] = features + [""] * (max_len - len(features))

        # Intersection of ALL methods
        common = set(method_features[list(method_features.keys())[0]])
        for v in method_features.values():
            common &= set(v)

        sheet_df["intersection_all"] = list(common) + [""] * (max_len - len(common))

        # Excel sheet name max = 31 chars
        sheet_df.to_excel(writer, sheet_name=network[:31], index=False)

print("\n🎉 Combined Excel file created successfully!")
print("📁 Saved at:", OUTPUT_FILE)
