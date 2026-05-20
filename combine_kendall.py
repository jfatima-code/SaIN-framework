# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 17:13:07 2025

@author: Javaria
"""

import os
import pandas as pd

# ==========================
# CONFIG
# ==========================
BASE_OUTPUTS_DIR = r"E:\New folder\Attributes\all\feature\Attribute_Paper\outputs_final_last"


# ==========================
# HELPERS
# ==========================

def load_any(path):
    ext = os.path.splitext(path)[1].lower()
    return pd.read_excel(path) if ext == ".xlsx" else pd.read_csv(path)

def norm(c):
    return c.lower().strip().replace(" ", "_")

# ✅ Proposed + method files dono detect honge (filename + columns)
def is_valid_kendall_file(fname, df):
    f = fname.lower()

    # ❌ skip ghalat combined files
    # NOTE: 'summary' HATA DIYA hai taa-ke 'kendall_tau_summary_results.csv' include ho
    bad = ["wide", "combined", "all_methods", "distinct", "jaccard", "lcc"]
    if any(b in f for b in bad):
        return False

    if not (f.endswith(".csv") or f.endswith(".xlsx")):
        return False

    # ✅ accept if ANY column has kendall/tau
    for c in df.columns:
        if "kendall" in c.lower() or "tau" in c.lower():
            return True

    # ✅ or filename has kendall/tau
    if "kendall" in f or "tau" in f:
        return True

    return False


# ==========================
# MAIN COMBINER
# ==========================

def combine_kendall_for_all_networks():

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

        # ✅ Walk network + all Method_* folders
        for root, dirs, files in os.walk(net_folder):

            # ✅ Detect method
            if os.path.abspath(root) == os.path.abspath(net_folder):
                method = "Proposed"
            else:
                method = os.path.basename(root)

            for fname in files:

                full_path = os.path.join(root, fname)

                try:
                    df = load_any(full_path)
                except Exception:
                    continue

                if not is_valid_kendall_file(fname, df):
                    continue

                print("✅ FOUND KENDALL FILE:", full_path)
                print("   ➤ Method:", method)

                # ---------- find lambda column ----------
                lambda_col = None
                for c in df.columns:
                    if "lambda" in norm(c) or "lamda" in norm(c):
                        lambda_col = c
                        break

                if lambda_col is None:
                    lambda_col = df.columns[0]

                # ✅ numeric + ROUND
                df[lambda_col] = pd.to_numeric(df[lambda_col], errors="coerce").round(4)

                # ============================================================
                # ✅ CASE 1: Normal Methods (Proposed, Method_D_Topsis, etc.)
                # ============================================================
                if "jaccard_centralities" not in norm(method):

                    val_col = None
                    for c in df.columns:
                        if c == lambda_col:
                            continue
                        if "kendall" in norm(c) or "tau" in norm(c):
                            val_col = c
                            break

                    if val_col is None:
                        val_col = df.columns[1]

                    out = df[[lambda_col, val_col]].copy()
                    out.columns = ["lambda", method]

                    if df_final is None:
                        df_final = out
                    else:
                        if method in df_final.columns:
                            df_final = pd.merge(
                                df_final,
                                out,
                                on="lambda",
                                how="outer",
                                suffixes=("", "_dup"),
                            )
                            df_final.drop(
                                columns=[c for c in df_final.columns if c.endswith("_dup")],
                                inplace=True
                            )
                        else:
                            df_final = pd.merge(df_final, out, on="lambda", how="outer")

                # ============================================================
                # ✅ CASE 2: Method_Jaccard_Centralities (ROW-WISE → WIDE FIX)
                # ============================================================
                else:
                    cent_col = None
                    tau_col = None

                    for c in df.columns:
                        if "centrality" in norm(c):
                            cent_col = c
                        if "kendall" in norm(c) or "tau" in norm(c):
                            tau_col = c

                    if cent_col is None or tau_col is None:
                        print("⚠️ Centrality/Tau columns not found in:", full_path)
                        continue

                    # ✅ PIVOT → one row per lambda, one column per centrality
                    piv = df.pivot_table(
                        index=lambda_col,
                        columns=cent_col,
                        values=tau_col,
                        aggfunc="mean"
                    ).reset_index()

                    # ✅ Direct centrality names (NO JaccardCent__)
                    piv.columns = ["lambda"] + list(piv.columns[1:])

                    if df_final is None:
                        df_final = piv
                    else:
                        df_final = pd.merge(df_final, piv, on="lambda", how="outer")

        # ==========================
        # ✅ FINAL SAVE PER NETWORK
        # ==========================
        if df_final is not None:

            df_final["lambda"] = pd.to_numeric(df_final["lambda"], errors="coerce")
            df_final = df_final.sort_values("lambda")

            out_path = os.path.join(
                net_folder, f"{net_name}_KENDALL_ALL_METHODS.xlsx"
            )

            df_final.to_excel(out_path, index=False)

            print("✅ FINAL KENDALL FILE SAVED:", out_path)

        else:
            print("⚠️ No valid Kendall files found for this network.")

    print("\n=== ✅ KENDALL COMBINATION DONE ===")


if __name__ == "__main__":
    combine_kendall_for_all_networks()
