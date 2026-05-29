# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 18:21:43 2025
@author: Javaria
"""

import os
import pandas as pd

BASE_OUTPUTS_DIR = r".......\outputs"


def load_any(path):
    ext = os.path.splitext(path)[1].lower()
    return pd.read_excel(path) if ext == ".xlsx" else pd.read_csv(path)


def norm(s):
    return s.lower().strip().replace(" ", "_")


def is_distinct_stats_file(fname):
    f = fname.lower()
    if "distinct_rank" not in f:
        return False
    bad = ["combined", "all_methods", "wide"]
    if any(b in f for b in bad):
        return False
    return f.endswith(".csv") or f.endswith(".xlsx")


def detect_centrality_from_filename(fname):
    f = fname.lower()
    if "degree" in f:
        return "Degree_Centrality"
    if "closeness" in f:
        return "Closeness_Centrality"
    if "betweenness" in f:
        return "Betweenness_Centrality"
    if "eigenvector" in f:
        return "Eigenvector_Centrality"
    if "pagerank" in f:
        return "PageRank_Centrality"
    return None


def combine_distinct_rank_all_in_one():

    networks = [
        os.path.join(BASE_OUTPUTS_DIR, d)
        for d in os.listdir(BASE_OUTPUTS_DIR)
        if os.path.isdir(os.path.join(BASE_OUTPUTS_DIR, d))
    ]

    for net_folder in sorted(networks):

        net_name = os.path.basename(net_folder)
        print("\n===================================")
        print("NETWORK:", net_name)

        out_path = os.path.join(
            net_folder, f"{net_name}_DISTINCT_RANK_ALL_IN_ONE.xlsx"
        )

        # ✅ AUTO-UPDATE MODE
        if os.path.exists(out_path):
            df_final = pd.read_excel(out_path)
            print("  🔁 Existing output file loaded for UPDATE")
        else:
            df_final = pd.DataFrame(columns=["name", "ratio"])
            print("  🆕 New output file will be created")

        for root, dirs, files in os.walk(net_folder):

            for fname in files:
                if not is_distinct_stats_file(fname):
                    continue

                full_path = os.path.join(root, fname)
                print("  📄", full_path)

                try:
                    df = load_any(full_path)
                except Exception as e:
                    print("    ❌ Load fail:", e)
                    continue

                # ✅ Detect ratio column
                ratio_col = None
                for c in df.columns:
                    if "ratio" in norm(c):
                        ratio_col = c
                        break

                if ratio_col is None:
                    print("    ⚠️ No ratio column found")
                    continue

                ratios = df[ratio_col].dropna().astype(float).values
                if len(ratios) == 0:
                    print("    ⚠️ Ratio column empty")
                    continue

                # ✅ EXACT VALUE (NO MEAN)
                ratio_val = float(ratios[0])

                # ✅ 🔒 SAFE NAME DETECTION
                # Case 1: Root distinct file → Proposed ONLY
                if (
                    os.path.abspath(root) == os.path.abspath(net_folder)
                    and norm(fname).startswith("distinct_rank")
                ):
                    name = "Proposed"

                # Case 2: Centralities inside jaccard folder
                else:
                    centrality_name = detect_centrality_from_filename(fname)
                    if "jaccard_centralities" in norm(root) and centrality_name is not None:
                        name = centrality_name
                    else:
                        # Case 3: Normal method = folder name
                        name = os.path.basename(root)

                # ✅ UPDATE IF EXISTS, ELSE ADD
                if name in df_final["name"].values:
                    df_final.loc[df_final["name"] == name, "ratio"] = ratio_val
                    print("    🔁 Updated:", name, "=", ratio_val)
                else:
                    df_final = pd.concat([
                        df_final,
                        pd.DataFrame([{"name": name, "ratio": ratio_val}])
                    ], ignore_index=True)
                    print("    ➕ Added:", name, "=", ratio_val)

        # ✅ SAVE UPDATED FILE
        df_final.to_excel(out_path, index=False)
        print("  ✅ FINAL UPDATED DISTINCT FILE SAVED:", out_path)

    print("\n=== ✅ DISTINCT ALL-IN-ONE AUTO-UPDATE DONE ===")


if __name__ == "__main__":
    combine_distinct_rank_all_in_one()
