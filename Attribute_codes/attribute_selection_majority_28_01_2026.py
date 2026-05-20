# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 07:42:25 2026

@author: Javaria
"""

import os
import pandas as pd
from collections import Counter

# ================= USER INPUT =================
ROOT_DIR = r"..\outputs_final_last"

METHODS = {
    "laplacian": "selected_features_lap.xlsx",
    "mcfs": "selected_features_mcfs.xlsx",
    "spec": "selected_features_spec.xlsx",
    "pca": "selected_features_pca.xlsx",
    "udfs": "selected_features_udfsBB.xlsx",
}

OUTPUT_FILE = os.path.join(
    ROOT_DIR, "All_Networks_Features_with_majority.xlsx"
)
# ==============================================


def get_feature_names(file_path):
    """Read column headers as selected features."""
    df = pd.read_excel(file_path)
    return list(df.columns)


with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    for network in os.listdir(ROOT_DIR):
        network_path = os.path.join(ROOT_DIR, network)

        if not os.path.isdir(network_path):
            continue

        print(f"📡 Processing network: {network}")

        method_features = {}

        # ---------- Read features per method ----------
        for method, fname in METHODS.items():
            file_path = os.path.join(network_path, method, fname)
            if os.path.exists(file_path):
                method_features[method] = get_feature_names(file_path)

        if not method_features:
            continue

        # ---------- Majority voting (CORRECT: per method) ----------
        feature_counts = Counter()

        for feats in method_features.values():
            for f in set(feats):        # 🔑 count once per method
                feature_counts[f] += 1

        selected_by_3 = [
            f for f, c in feature_counts.items() if c >= 3
        ]

        # ---------- Intersection (all methods) ----------
        common = set(list(method_features.values())[0])
        for feats in method_features.values():
            common &= set(feats)

        # ---------- Build DataFrame (simple & safe) ----------
        data = {}

        for method, feats in method_features.items():
            data[method] = pd.Series(feats)

        data["intersection_all"] = pd.Series(list(common))
        data["selected_by_3_methods"] = pd.Series(selected_by_3)

        sheet_df = pd.DataFrame(data)

        # ---------- Save ----------
        sheet_df.to_excel(
            writer, sheet_name=network[:31], index=False
        )

print("\n🎉 DONE! Excel file created successfully.")
print("📁 Saved at:", OUTPUT_FILE)
