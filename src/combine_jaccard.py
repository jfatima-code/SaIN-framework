# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 16:16:57 2025

@author: Javaria
"""

import os
import pandas as pd

BASE_OUTPUTS_DIR = r"E:\New folder\Attributes\all\feature\Attribute_Paper\outputs_final_last"

def load_any(path):
    ext = os.path.splitext(path)[1].lower()
    return pd.read_excel(path) if ext == ".xlsx" else pd.read_csv(path)

def norm(c):
    return c.lower().strip().replace(" ", "_")

def is_valid_jaccard_filename(fname):
    f = fname.lower()

    # ✅ must contain these
    if "jaccard" not in f or "similar" not in f:
        return False

    # ❌ skip combined / summary / wide files
    bad = ["wide", "summary", "combined", "all_methods", "distinct", "kendall", "lcc"]
    if any(b in f for b in bad):
        return False

    # ✅ must be csv/xlsx
    return f.endswith(".csv") or f.endswith(".xlsx")

def combine_jaccard_for_all_networks():

    networks = [
        os.path.join(BASE_OUTPUTS_DIR, d)
        for d in os.listdir(BASE_OUTPUTS_DIR)
        if os.path.isdir(os.path.join(BASE_OUTPUTS_DIR, d))
    ]

    for net_folder in sorted(networks):
        net_name = os.path.basename(net_folder)
        print("\n===================================")
        print("NETWORK:", net_name)

        df_final = None

        # ✅ walk inside network + method folders
        for root, dirs, files in os.walk(net_folder):

            # method name
            if os.path.abspath(root) == os.path.abspath(net_folder):
                method = "Proposed"
            else:
                method = os.path.basename(root)

            for fname in files:

                # ✅ flexible filename matching
                if not is_valid_jaccard_filename(fname):
                    continue

                full_path = os.path.join(root, fname)

                print("✅ FOUND JACCARD FILE:", full_path)
                print("   ➤ Method:", method)

                try:
                    df = load_any(full_path)
                except Exception as e:
                    print("❌ Failed to load:", full_path, "->", e)
                    continue

                # ---------- find top_nodes column ----------
                top_col = None
                for c in df.columns:
                    if any(k in norm(c) for k in ["top_nodes", "top", "k"]):
                        top_col = c
                        break

                if top_col is None:
                    print("❌ top_nodes / k column not found in:", fname)
                    continue

                # ✅ numeric conversion (safe sorting)
                df[top_col] = pd.to_numeric(df[top_col], errors="coerce")

                # ---------- find jaccard value columns ----------
                jac_cols = [
                    c for c in df.columns
                    if c != top_col and ("jaccard" in norm(c) or "jac" in norm(c))
                ]

                # ✅ fallback: take any 1 column except top_nodes
                if not jac_cols:
                    jac_cols = [c for c in df.columns if c != top_col]

                # ---------- process each jaccard column ----------
                for col in jac_cols:

                    # centralities folder special naming
                    if "jaccard_centralities" in norm(method):
                        new_col = f"JaccardCent__{col}"
                    else:
                        new_col = method

                    out = df[[top_col, col]].copy()
                    out.columns = ["top_nodes", new_col]

                    if df_final is None:
                        df_final = out
                    else:
                        if new_col in df_final.columns:
                            df_final = pd.merge(
                                df_final,
                                out,
                                on="top_nodes",
                                how="outer",
                                suffixes=("", "_dup"),
                            )
                            df_final.drop(
                                columns=[c for c in df_final.columns if c.endswith("_dup")],
                                inplace=True
                            )
                        else:
                            df_final = pd.merge(df_final, out, on="top_nodes", how="outer")

        # ---------- FINAL SAVE ----------
        if df_final is not None:

            df_final["top_nodes"] = pd.to_numeric(df_final["top_nodes"], errors="coerce")
            df_final = df_final.sort_values("top_nodes")

            out_path = os.path.join(
                net_folder, f"{net_name}_JACCARD_ALL_METHODS.xlsx"
            )
            df_final.to_excel(out_path, index=False)

            print("✅ FINAL SAVED:", out_path)

        else:
            print("⚠️ No matching Jaccard similarity files found for this network.")

    print("\n=== ✅ JACCARD COMBINATION DONE ===")

if __name__ == "__main__":
    combine_jaccard_for_all_networks()
