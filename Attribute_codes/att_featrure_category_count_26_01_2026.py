# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 00:38:39 2026

@author: Javaria
"""

import os
import pandas as pd

# ================== PATHS ==================
COMBINED_EXCEL = r"All_Networks_Features.xlsx"

CATEGORY_FILE = r"feature_classes_as_per_code_25_01_2026.xlsx"

OUTPUT_EXCEL = r"All_Networks_Category_Counts_9_4_26.xlsx"

METHODS = ["laplacian", "mcfs", "spec", "pca", "udfs"]
# ==========================================
import re

def normalize_feature(s):
    s = str(s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


# --------- Load category mapping (2-column format) ----------
cat_df = pd.read_excel(CATEGORY_FILE)

# If your file has no headers, pandas will name them 0,1 automatically OR "Unnamed: 0"
# We'll just take first two columns safely:
cat_df = pd.read_excel(CATEGORY_FILE, header=None)
cat_df = cat_df.iloc[:, :2]
cat_df.columns = ["Feature_Name", "Category"]


cat_df["Feature_Name"] = cat_df["Feature_Name"].apply(normalize_feature)

print("\n🔎 Raw feature names from category file (first 50):")
print(cat_df["Feature_Name"].head(50).tolist())

# Build mapping dict
feature_to_cat = dict(zip(cat_df["Feature_Name"], cat_df["Category"]))
all_categories = sorted(cat_df["Category"].dropna().unique().tolist())


# --------- Process combined excel (each sheet = network) ----------
xl = pd.ExcelFile(COMBINED_EXCEL)

rows = []
details_rows = []

for sheet in xl.sheet_names:
    network = sheet
    df = xl.parse(sheet)

    for method in METHODS:
        if method not in df.columns:
            continue

        feats = df[method].dropna().apply(normalize_feature)



        feats = feats[feats != ""]

        cats = feats.map(lambda f: feature_to_cat.get(f, "Unknown"))
        counts = cats.value_counts().to_dict()
        print("Exists in mapping:",
              "lgc scores" in feature_to_cat,
              "isolating centrality" in feature_to_cat,
              "hvgc index" in feature_to_cat)

        row = {
            "Network": network,
            "Method": method,
            "Total": int(len(feats)),
            "Unknown": int(counts.get("Unknown", 0)),
        }

        # Add all category columns
        for c in all_categories:
            row[c] = int(counts.get(c, 0))

        rows.append(row)

        # Details (optional but very useful)
        for f, c in zip(feats.tolist(), cats.tolist()):
            details_rows.append({
                "Network": network,
                "Method": method,
                "Feature_Name": f,
                "Category": c
            })

summary_df = pd.DataFrame(rows)

# Ensure all category columns exist even if 0
for c in all_categories:
    if c not in summary_df.columns:
        summary_df[c] = 0

# Nice column order
summary_df = summary_df[["Network", "Method"] + all_categories + ["Unknown", "Total"]]
details_df = pd.DataFrame(details_rows)
# --------- Extract Unknown features ----------
unknown_df = details_df[details_df["Category"] == "Unknown"]

print("\n🔍 Unknown Features:\n")
print(unknown_df)

# --------- Save output ----------
with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Counts", index=False)
    details_df.to_excel(writer, sheet_name="Details", index=False)

print("✅ Saved:", OUTPUT_EXCEL)
print("Exists in mapping:",
      "isolating centrality" in feature_to_cat)
