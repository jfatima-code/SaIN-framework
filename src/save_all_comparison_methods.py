# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 15:26:27 2025

@author: Javaria
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 15:26:27 2025

@author: Javaria
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kendalltau

# ==========================
# CONFIGURATION
# ==========================

BASE_OUTPUTS_DIR = r"...outputs_final_last"
ATTR_DIR         = r"....\Attribute_codes"

# Method subfolders inside each network folder
METHOD1_SUBFOLDER = "Method_Jaccard_Centralities"
METHOD2_SUBFOLDER = "Method_Random"
METHOD3_SUBFOLDER = "Method_D_Topsis"
METHOD4_SUBFOLDER = "Method_WSM"

# Centralities for Method 1
CENTRALITY_COLUMNS = [
    "Degree Centrality",
    "Betweenness Centrality",
    "PageRank Centrality",
    "Closeness Centrality",
    "Eigenvector Centrality",
]

# Centralities for D-TOPSIS / WSM methods
D_TOPSIS_COLUMNS = [
    "Degree Centrality",
    "Betweenness Centrality",
    "Closeness Centrality",
]

# ==========================
# HELPER FUNCTIONS
# ==========================

def normalize_name(name: str) -> str:
    """Lowercase + remove spaces + underscores."""
    return str(name).lower().replace(" ", "").replace("_", "")


def find_column(df: pd.DataFrame, target_name: str):
    """
    It finds column in DataFrame that matches target_name 
    (case-insensitive, spaces/underscore ignore).
    """
    target_norm = normalize_name(target_name)
    for col in df.columns:
        if normalize_name(col) == target_norm:
            return col
    return None


def find_attribute_file(network_name: str):
    """
    Finding the attribute file:

    first exact:
        graph_attributes_<network_name>_check.csv
    if it couldnt find exact then safer flexible search:
    
      - starts form 'graph_attributes_' 
      - middle part (network name) should match EXACT 
        (will not mix g10 ko g102 )
    """
    # 1) Exact expected name
    exact = os.path.join(
        ATTR_DIR,
        f"graph_attributes_{network_name}_check.csv"
    )
    if os.path.isfile(exact):
        return exact

    # 2) Safer flexible search
    net_low = network_name.lower()

    for fname in os.listdir(ATTR_DIR):
        low = fname.lower()
        if not low.endswith(".csv"):
            continue
        if not low.startswith("graph_attributes_"):
            continue

        middle = low[len("graph_attributes_"):]
        if middle.endswith(".csv"):
            middle = middle[:-4]
        if middle.endswith("_check"):
            middle = middle[:-6]

        if middle == net_low:
            return os.path.join(ATTR_DIR, fname)

    return None


def find_baseline_spreading_file(network_dir: str):
    
    if not os.path.isdir(network_dir):
        return None

    csv_files = [f for f in os.listdir(network_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        return None

    # Prefer files with 'spread' + ('average' or 'avg') and *no* 'lambda'
    for fname in csv_files:
        low = fname.lower()
        if "spread" in low and ("average" in low or "avg" in low) and "lambda" not in low:
            return os.path.join(network_dir, fname)

    # Fallback: any spread without 'spread' (without lambda)
    for fname in csv_files:
        low = fname.lower()
        if "spread" in low and "lambda" not in low:
            return os.path.join(network_dir, fname)

    return None


def auto_kendall_for_lambdas(
    network_dir: str,
    df_node_closeness: pd.DataFrame,
    df_spread: pd.DataFrame,
    spread_val_col: str,
    method_out_dir: str
):
    
    kendall_records = []

    prefix = "average_spreading_abilities_lambda_"
    lambda_files = []

    for fname in os.listdir(network_dir):
        low = fname.lower()
        if not low.endswith(".csv"):
            continue
        if not low.startswith(prefix):
            continue

        num_part = fname[len(prefix):-4]
        try:
            lam_val = float(num_part)
            full_path = os.path.join(network_dir, fname)
            lambda_files.append((lam_val, full_path))
        except ValueError:
            continue

    if not lambda_files:
        print("  [Info] No average_spreading_abilities_lambda_*.csv files found for this network. Skipping Kendall Tau.")
        kendall_df = pd.DataFrame(columns=["Lambda", "Kendall_Tau", "P_Value"])
        kendall_out_path = os.path.join(method_out_dir, "kendall_tau_results.csv")
        kendall_df.to_csv(kendall_out_path, index=False)
        print(f"  Saved empty Kendall Tau results: {kendall_out_path}")
        return

    lambda_files.sort(key=lambda x: x[0])

    for lam_val, lam_path in lambda_files:
        avg_lam = pd.read_csv(lam_path)

        avg_lam_node_col = find_column(avg_lam, "Starting Node") or find_column(avg_lam, "Node")
        avg_lam_spread_col = find_column(avg_lam, "Average Spreading Ability")

        if avg_lam_node_col is None or avg_lam_spread_col is None:
            print(f"  [Warning] Lambda {lam_val:.6f}: required columns not found. Skipping.")
            kendall_records.append({"Lambda": lam_val, "Kendall_Tau": None, "P_Value": None})
            continue

        avg_lam["Node"] = avg_lam[avg_lam_node_col]
        avg_lam["Rank"] = avg_lam[avg_lam_spread_col].rank(method="dense", ascending=False)

        avg_lam_common = pd.merge(
            avg_lam[["Node", "Rank"]],
            df_node_closeness[["Node"]],
            on="Node",
            how="inner",
        )

        infl_common = pd.merge(
            df_node_closeness,
            avg_lam_common[["Node"]],
            on="Node",
            how="inner",
        )
        infl_common["Rank_node_infl"] = infl_common["Relative Closeness"].rank(
            method="dense", ascending=False
        )

        merged = pd.merge(
            avg_lam_common,
            infl_common[["Node", "Rank_node_infl"]],
            on="Node",
            how="inner",
        )

        if merged.empty or merged["Rank"].nunique() <= 1 or merged["Rank_node_infl"].nunique() <= 1:
            print(f"  [Info] Lambda {lam_val:.6f}: insufficient variation for Kendall Tau.")
            tau, pval = None, None
        else:
            tau, pval = kendalltau(merged["Rank"], merged["Rank_node_infl"])
            print(f"  Lambda {lam_val:.6f}: Kendall Tau = {tau}, p = {pval}")

        kendall_records.append({"Lambda": lam_val, "Kendall_Tau": tau, "P_Value": pval})

    kendall_df = pd.DataFrame(kendall_records)
    kendall_out_path = os.path.join(method_out_dir, "kendall_tau_results.csv")
    kendall_df.to_csv(kendall_out_path, index=False)
    print(f"  Saved Kendall Tau results: {kendall_out_path}")


def jaccard_and_rank_common(
    df_spread: pd.DataFrame,
    spread_val_col: str,
    spread_node_col: str,
    df_node_closeness: pd.DataFrame,
    method_out_dir: str,
    label: str = "Method"
):
    
    avg_for_rank = pd.merge(
        df_spread,
        df_node_closeness[["Node"]],
        left_on=spread_node_col,
        right_on="Node",
        how="inner",
    )

    avg_for_rank["Average Spreading Ability"] = avg_for_rank[spread_val_col]
    avg_for_rank["Rank"] = avg_for_rank["Average Spreading Ability"].rank(
        method="dense", ascending=False
    )

    infl_sorted = df_node_closeness.sort_values(by="Relative Closeness", ascending=False).copy()
    infl_sorted["Rank"] = infl_sorted["Relative Closeness"].rank(method="dense", ascending=False)

    avg_rank_path = os.path.join(method_out_dir, "average_spreading_ranking.csv")
    infl_rank_path = os.path.join(method_out_dir, "node_influence_ranking.csv")
    avg_for_rank.to_csv(avg_rank_path, index=False)
    infl_sorted.to_csv(infl_rank_path, index=False)
    print(f"  [{label}] Saved rankings: {avg_rank_path}, {infl_rank_path}")

    spread_node_for_rank = find_column(avg_for_rank, "Starting Node") or find_column(avg_for_rank, "Node")
    if spread_node_for_rank is None:
        print(f"  [Warning] [{label}] No node column in baseline spreading for Jaccard. Skipping Jaccard.")
    else:
        jaccard_rows = []
        for top_n in range(5, 81, 5):
            row = {"Top_Nodes": top_n}
            spread_top = set(
                avg_for_rank[avg_for_rank["Rank"] <= top_n][spread_node_for_rank].tolist()
            )
            infl_top = set(infl_sorted[infl_sorted["Rank"] <= top_n]["Node"].tolist())

            union = spread_top.union(infl_top)
            inter = spread_top.intersection(infl_top)
            jacc = len(inter) / len(union) if len(union) > 0 else 0.0
            row[label] = jacc
            jaccard_rows.append(row)

        jaccard_df = pd.DataFrame(jaccard_rows)
        jaccard_out_path = os.path.join(method_out_dir, "jaccard_similarity_results_top_5_to_80.csv")
        jaccard_df.to_csv(jaccard_out_path, index=False)
        print(f"  [{label}] Saved Jaccard results: {jaccard_out_path}")

    return avg_for_rank, infl_sorted


# ==========================
# METHOD 1:
# Centrality vs Spreading – Jaccard (top-5..80)
# + Centrality vs Lambda – Kendall
# ==========================

def run_jaccard_centralities_for_network(network_name: str):
    print(f"\n=== [Method 1] Network: {network_name} ===")

    network_dir = os.path.join(BASE_OUTPUTS_DIR, network_name)
    if not os.path.isdir(network_dir):
        print(f"  [Skip] {network_dir} is not a folder.")
        return

    # Attribute & spreading files
    attr_file = find_attribute_file(network_name)
    if attr_file is None:
        print("  [Skip] No attribute CSV found matching pattern for this network.")
        return

    spreading_file = find_baseline_spreading_file(network_dir)
    if spreading_file is None:
        print("  [Skip] No baseline spreading CSV found in this network folder.")
        return

    print(f"  Using attributes: {attr_file}")
    print(f"  Using spreading:  {spreading_file}")

    method_out_dir = os.path.join(network_dir, METHOD1_SUBFOLDER)
    os.makedirs(method_out_dir, exist_ok=True)

    df_attr = pd.read_csv(attr_file)
    df_spread = pd.read_csv(spreading_file)

    node_col = find_column(df_attr, "Node")
    if node_col is None:
        print("  [Error] Could not find 'Node' column in attributes. Skipping.")
        return

    # ---- Rank centralities ----
    ranked_centralities = {}

    for col_target in CENTRALITY_COLUMNS:
        real_col = find_column(df_attr, col_target)
        if real_col is None:
            print(f"  [Warning] Centrality like '{col_target}' not found. Skipping this one.")
            continue

        tmp = df_attr[[node_col, real_col]].copy()
        tmp = tmp.sort_values(by=real_col, ascending=False)
        tmp.rename(columns={node_col: "Node", real_col: "Score"}, inplace=True)
        tmp["Rank"] = tmp["Score"].rank(ascending=False, method="dense")

        ranked_centralities[col_target] = tmp

        out_path = os.path.join(method_out_dir, f"{col_target}_ranks.csv")
        tmp.to_csv(out_path, index=False)
        print(f"  Saved ranks: {out_path}")

    if not ranked_centralities:
        print("  [Error] No centrality columns found. Skipping Jaccard/Kendall.")
        return

    # ---- Rank baseline spreading ability ----
    spread_val_col = find_column(df_spread, "Average Spreading Ability")
    spread_node_col = find_column(df_spread, "Starting Node") or find_column(df_spread, "Node")

    if spread_val_col is None or spread_node_col is None:
        print("  [Error] Could not find spreading columns (value or node). Skipping.")
        return

    df_spread_sorted = df_spread.sort_values(by=spread_val_col, ascending=False).copy()
    df_spread_sorted["Rank"] = df_spread_sorted[spread_val_col].rank(
        ascending=False, method="dense"
    )

    spreading_out_path = os.path.join(method_out_dir, "Average_Spreading_Ability_ranks.csv")
    df_spread_sorted.to_csv(spreading_out_path, index=False)
    print(f"  Saved spreading ranks: {spreading_out_path}")

    # ---- Jaccard similarity top-5..80 (centrality vs baseline spreading) ----
    jaccard_results = []

    for top_n in range(5, 81, 5):
        row = {"Top_Nodes": top_n}

        spread_top_nodes = set(
            df_spread_sorted[df_spread_sorted["Rank"] <= top_n][spread_node_col].tolist()
        )

        for col_target, df_ranked in ranked_centralities.items():
            central_top_nodes = set(
                df_ranked[df_ranked["Rank"] <= top_n]["Node"].tolist()
            )

            union = spread_top_nodes.union(central_top_nodes)
            inter = spread_top_nodes.intersection(central_top_nodes)
            jaccard = len(inter) / len(union) if len(union) > 0 else 0.0

            output_col_name = col_target.replace(" ", "_")
            row[output_col_name] = jaccard

        jaccard_results.append(row)

    jaccard_df = pd.DataFrame(jaccard_results)
    jaccard_out_path = os.path.join(method_out_dir, "jaccard_similarity_results_top_5_to_80.csv")
    jaccard_df.to_csv(jaccard_out_path, index=False)
    print(f"  Saved Jaccard results: {jaccard_out_path}")

    # ======================================================
    # NEW: Centrality vs Lambda spreading — Kendall Tau
    # ======================================================
    prefix = "average_spreading_abilities_lambda_"
    lambda_files = []

    for fname in os.listdir(network_dir):
        low = fname.lower()
        if not low.endswith(".csv"):
            continue
        if not low.startswith(prefix):
            continue

        num_part = fname[len(prefix):-4]
        try:
            lam_val = float(num_part)
            full_path = os.path.join(network_dir, fname)
            lambda_files.append((lam_val, full_path))
        except ValueError:
            continue

    if not lambda_files:
        print("  [Info] No lambda-based spreading files found for centrality Kendall.")
        return

    lambda_files.sort(key=lambda x: x[0])

    # centrality_dfs: {centrality_name: DataFrame(Node, Rank_centrality)}
    centrality_dfs = {}
    for col_target, df_ranked in ranked_centralities.items():
        df_cent = df_ranked[["Node", "Rank"]].copy()
        df_cent.rename(columns={"Rank": "Rank_centrality"}, inplace=True)
        cent_key = col_target.strip().replace(" ", "_")
        centrality_dfs[cent_key] = df_cent

    kendall_records = []

    for lam_val, lam_path in lambda_files:
        df_lam = pd.read_csv(lam_path)

        lam_node_col = find_column(df_lam, "Starting Node") or find_column(df_lam, "Node")
        lam_spread_col = find_column(df_lam, "Average Spreading Ability")

        if lam_node_col is None or lam_spread_col is None:
            print(f"  [Warning] Lambda {lam_val:.6f}: could not find node or spreading columns. Skipping this lambda.")
            continue

        df_lam_rank = df_lam[[lam_node_col, lam_spread_col]].copy()
        df_lam_rank.rename(columns={lam_node_col: "Node", lam_spread_col: "Spread"}, inplace=True)
        df_lam_rank["Rank_lambda"] = df_lam_rank["Spread"].rank(method="dense", ascending=False)

        for cent_key, df_cent in centrality_dfs.items():
            merged = pd.merge(
                df_cent,
                df_lam_rank[["Node", "Rank_lambda"]],
                on="Node",
                how="inner",
            )

            if merged.empty:
                print(f"    [Info] Lambda {lam_val:.6f}, centrality {cent_key}: no common nodes. Skipping.")
                kendall_records.append({
                    "Lambda": lam_val,
                    "Centrality": cent_key,
                    "Kendall_Tau": None,
                    "P_Value": None
                })
                continue

            if merged["Rank_centrality"].nunique() <= 1 or merged["Rank_lambda"].nunique() <= 1:
                print(f"    [Info] Lambda {lam_val:.6f}, {cent_key}: insufficient variation. Tau undefined.")
                tau, pval = None, None
            else:
                tau, pval = kendalltau(merged["Rank_centrality"], merged["Rank_lambda"])
                print(f"    [Kendall] Lambda {lam_val:.6f}, {cent_key}: Tau={tau}, p={pval}")

            kendall_records.append({
                "Lambda": lam_val,
                "Centrality": cent_key,
                "Kendall_Tau": tau,
                "P_Value": pval
            })

    if kendall_records:
        kendall_df = pd.DataFrame(kendall_records)
        kendall_out_path2 = os.path.join(method_out_dir, "kendall_tau_centralities_vs_lambda.csv")
        kendall_df.to_csv(kendall_out_path2, index=False)
        print(f"  Saved centrality vs lambda Kendall Tau: {kendall_out_path2}")
    else:
        print("  [Info] No Kendall records computed for centrality vs lambda.")


# ==========================
# METHOD 2: RANDOM
# ==========================

def run_random_method_for_network(network_name: str):
    print(f"\n=== [Method 2 - Random] Network: {network_name} ===")

    network_dir = os.path.join(BASE_OUTPUTS_DIR, network_name)
    if not os.path.isdir(network_dir):
        print(f"  [Skip] {network_dir} is not a folder.")
        return

    attr_file = find_attribute_file(network_name)
    if attr_file is None:
        print("  [Skip] No attribute CSV found matching pattern for this network.")
        return

    baseline_spread_file = find_baseline_spreading_file(network_dir)
    if baseline_spread_file is None:
        print("  [Skip] No baseline spreading CSV found for Random method.")
        return

    print(f"  Using attributes: {attr_file}")
    print(f"  Using baseline spreading:  {baseline_spread_file}")

    method_out_dir = os.path.join(network_dir, METHOD2_SUBFOLDER)
    os.makedirs(method_out_dir, exist_ok=True)

    df_attr = pd.read_csv(attr_file)
    df_spread = pd.read_csv(baseline_spread_file)

    node_col_attr = find_column(df_attr, "Node")
    if node_col_attr is None:
        print("  [Error] 'Node' column not found in attributes. Skipping.")
        return

    spread_val_col = find_column(df_spread, "Average Spreading Ability")
    spread_node_col = find_column(df_spread, "Starting Node") or find_column(df_spread, "Node")
    if spread_val_col is None or spread_node_col is None:
        print("  [Error] Spreading file does not have proper node/value columns. Skipping.")
        return

    all_cols = [c for c in df_attr.columns if normalize_name(c) != normalize_name("Node")]
    if not all_cols:
        print("  [Error] No attribute columns found (besides Node).")
        return

    n_select = min(25, len(all_cols)) #25
    random_columns = np.random.choice(all_cols, n_select, replace=False)

    attr_sel_cols = [node_col_attr] + list(random_columns)
    df_attr_sel = df_attr[attr_sel_cols].copy()

    random_attributes_data = df_attr_sel.drop(columns=[node_col_attr])
    random_attr_path = os.path.join(method_out_dir, "randomly_selected_attributes.csv")
    random_attributes_data.to_csv(random_attr_path, index=False)
    print(f"  Saved random attributes: {random_attr_path}")

    df_spread_ren = df_spread[[spread_node_col, spread_val_col]].copy()
    df_spread_ren.rename(
        columns={
            spread_node_col: node_col_attr,
            spread_val_col: "Average Spreading Ability",
        },
        inplace=True,
    )

    selected_data_1 = pd.merge(
        df_attr_sel,
        df_spread_ren,
        on=node_col_attr,
        how="inner"
    )

    if selected_data_1.empty:
        print("  [Error] No common nodes between attributes and spreading. Skipping.")
        return

    if len(selected_data_1) < len(df_attr_sel):
        print(f"  [Info] {len(df_attr_sel) - len(selected_data_1)} nodes dropped due to no spreading info.")

    cols_reordered = [node_col_attr] + [c for c in selected_data_1.columns if c != node_col_attr]
    selected_data_1 = selected_data_1[cols_reordered]

    selected_xlsx_path = os.path.join(method_out_dir, "selected_data_output.xlsx")
    selected_data_1.to_excel(selected_xlsx_path, index=False)
    print(f"  Saved selected_data_output.xlsx: {selected_xlsx_path}")

    scaler = MinMaxScaler()
    selected_data_normalized = selected_data_1.copy()

    columns_to_normalize = [c for c in selected_data_normalized.columns if c != node_col_attr]
    selected_data_normalized[columns_to_normalize] = scaler.fit_transform(
        selected_data_normalized[columns_to_normalize]
    )
    selected_data = selected_data_normalized

    spread_col_name = "Average Spreading Ability"

    df_without_spreading = selected_data.drop(columns=[spread_col_name])
    out_without_spread_xlsx = os.path.join(method_out_dir, "centrality_results_without_spreading.xlsx")
    df_without_spreading.to_excel(out_without_spread_xlsx, index=False)

    matrix_without_spreading = df_without_spreading.drop(columns=[node_col_attr]).values.copy()
    matrix_without_spreading_path = os.path.join(method_out_dir, "matrix_without_spreading.csv")
    pd.DataFrame(matrix_without_spreading).to_csv(matrix_without_spreading_path, index=False, header=None)

    df_with_spreading = selected_data.copy()
    out_with_spread_xlsx = os.path.join(method_out_dir, "centrality_results_with_spreading.xlsx")
    df_with_spreading.to_excel(out_with_spread_xlsx, index=False)

    matrix_with_spreading = df_with_spreading.drop(columns=[node_col_attr]).values
    matrix_with_spreading_path = os.path.join(method_out_dir, "matrix_with_spreading.csv")
    pd.DataFrame(matrix_with_spreading).to_csv(matrix_with_spreading_path, index=False, header=None)

    normalized_matrix_without_spreading = matrix_without_spreading
    norm_without_path = os.path.join(method_out_dir, "normalized_centrality_matrix_without_spreading.csv")
    np.savetxt(norm_without_path, normalized_matrix_without_spreading, delimiter=",")
    print(f"  [Random] Saved normalized matrix (without spreading): {norm_without_path}")

    normalized_matrix_with_spreading = matrix_with_spreading
    normalized_df = pd.DataFrame(
        normalized_matrix_with_spreading,
        columns=df_with_spreading.columns[1:],  # assume first is Node
    )
    norm_data_path = os.path.join(method_out_dir, "normalized_data.csv")
    normalized_df.to_csv(norm_data_path, index=False)

    df_1 = normalized_df.copy()
    gra_spread_col = find_column(df_1, "Average Spreading Ability")
    if gra_spread_col is None:
        print("  [Error] 'Average Spreading Ability' not found in normalized data. Skipping GRA.")
        return

    Y0 = df_1[gra_spread_col].values
    Yj = df_1.drop(columns=[gra_spread_col]).values

    a = np.min(np.abs(Y0[:, np.newaxis] - Yj))
    b = np.max(np.abs(Y0[:, np.newaxis] - Yj))
    rho = 0.5

    R = (a + rho * b) / (np.abs(Y0[:, np.newaxis] - Yj) + rho * b)
    grey_relational_degree = np.mean(R, axis=0)

    weights = np.array(grey_relational_degree) / np.sum(grey_relational_degree)

    weighted_matrix = weights * normalized_matrix_without_spreading
    weighted_matrix_df = pd.DataFrame(weighted_matrix)
    weighted_matrix_path = os.path.join(method_out_dir, "weighted_matrix.csv")
    weighted_matrix_df.to_csv(weighted_matrix_path, index=False, header=False)

    positive_ideal_object = np.max(weighted_matrix, axis=0)
    negative_ideal_object = np.min(weighted_matrix, axis=0)

    euclidean_distance_positive = np.sqrt(np.sum((weighted_matrix - positive_ideal_object) ** 2, axis=1))
    euclidean_distance_negative = np.sqrt(np.sum((weighted_matrix - negative_ideal_object) ** 2, axis=1))

    relative_closeness = euclidean_distance_negative / (
        euclidean_distance_positive + euclidean_distance_negative
    )

    nodes_series = selected_data_1[node_col_attr]
    node_closeness = list(zip(nodes_series, relative_closeness))

    df_node_closeness = pd.DataFrame(node_closeness, columns=["Node", "Relative Closeness"])
    node_influence_path = os.path.join(method_out_dir, "node_influence.csv")
    df_node_closeness.to_csv(node_influence_path, index=False)
    print(f"  Node influence saved: {node_influence_path}")

    avg_for_rank, infl_sorted = jaccard_and_rank_common(
        df_spread,
        spread_val_col,
        spread_node_col,
        df_node_closeness,
        method_out_dir,
        label="Random_Method",
    )

    auto_kendall_for_lambdas(
        network_dir=network_dir,
        df_node_closeness=df_node_closeness,
        df_spread=df_spread,
        spread_val_col=spread_val_col,
        method_out_dir=method_out_dir,
    )


# ==========================
# METHOD 3: D-TOPSIS
# ==========================

def run_d_topsis_method_for_network(network_name: str):
    print(f"\n=== [Method 3 - D_Topsis] Network: {network_name} ===")

    network_dir = os.path.join(BASE_OUTPUTS_DIR, network_name)
    if not os.path.isdir(network_dir):
        print(f"  [Skip] {network_dir} is not a folder.")
        return

    attr_file = find_attribute_file(network_name)
    if attr_file is None:
        print("  [Skip] No attribute CSV found matching pattern for this network.")
        return

    baseline_spread_file = find_baseline_spreading_file(network_dir)
    if baseline_spread_file is None:
        print("  [Skip] No baseline spreading CSV found for D_Topsis method.")
        return

    print(f"  Using attributes: {attr_file}")
    print(f"  Using baseline spreading:  {baseline_spread_file}")

    method_out_dir = os.path.join(network_dir, METHOD3_SUBFOLDER)
    os.makedirs(method_out_dir, exist_ok=True)

    df_attr = pd.read_csv(attr_file)
    df_spread = pd.read_csv(baseline_spread_file)

    node_col_attr = find_column(df_attr, "Node")
    if node_col_attr is None:
        print("  [Error] 'Node' column not found in attributes. Skipping.")
        return

    real_cols = {}
    for col in D_TOPSIS_COLUMNS:
        c = find_column(df_attr, col)
        if c is None:
            print(f"  [Error] D_Topsis centrality '{col}' not found in attributes. Skipping method for this network.")
            return
        real_cols[col] = c

    spread_val_col = find_column(df_spread, "Average Spreading Ability")
    spread_node_col = find_column(df_spread, "Starting Node") or find_column(df_spread, "Node")
    if spread_val_col is None or spread_node_col is None:
        print("  [Error] Spreading file does not have proper node/value columns. Skipping.")
        return

    selected_cols = [node_col_attr] + [real_cols[c] for c in D_TOPSIS_COLUMNS]
    df_attr_sel = df_attr[selected_cols].copy()

    df_spread_ren = df_spread[[spread_node_col, spread_val_col]].copy()
    df_spread_ren.rename(
        columns={
            spread_node_col: node_col_attr,
            spread_val_col: "Average Spreading Ability",
        },
        inplace=True,
    )

    final_df = pd.merge(
        df_attr_sel,
        df_spread_ren,
        on=node_col_attr,
        how="inner"
    )

    if final_df.empty:
        print("  [Error] No common nodes between attributes and spreading. Skipping.")
        return

    if len(final_df) < len(df_attr_sel):
        print(f"  [Info] {len(df_attr_sel) - len(final_df)} nodes dropped due to no spreading info.")

    df_with_spreading = final_df.copy()
    cent_cols_only = [node_col_attr] + [real_cols[c] for c in D_TOPSIS_COLUMNS]
    df_without_spreading = final_df[cent_cols_only].copy()

    out_without_spread_xlsx = os.path.join(method_out_dir, "centrality_results_without_spreading.xlsx")
    out_with_spread_xlsx = os.path.join(method_out_dir, "centrality_results_with_spreading.xlsx")
    df_without_spreading.to_excel(out_without_spread_xlsx, index=False)
    df_with_spreading.to_excel(out_with_spread_xlsx, index=False)
    print(f"  Saved D_Topsis Excel: {out_without_spread_xlsx}, {out_with_spread_xlsx}")

    matrix_without_spreading = df_without_spreading.drop(columns=[node_col_attr]).values.copy()
    matrix_with_spreading = df_with_spreading.drop(columns=[node_col_attr]).values

    col_norm = np.sqrt((matrix_without_spreading ** 2).sum(axis=0))
    col_norm[col_norm == 0] = 1.0
    normalized_matrix_without_spreading = matrix_without_spreading / col_norm

    norm_without_path = os.path.join(method_out_dir, "normalized_centrality_matrix_without_spreading.csv")
    np.savetxt(norm_without_path, normalized_matrix_without_spreading, delimiter=",")
    print(f"  Saved normalized matrix (without spreading): {norm_without_path}")

    col_min = matrix_with_spreading.min(axis=0)
    col_sum = matrix_with_spreading.sum(axis=0)
    col_sum[col_sum == 0] = 1.0
    normalized_matrix_with_spreading = (matrix_with_spreading - col_min) / col_sum

    normalized_df = pd.DataFrame(
        normalized_matrix_with_spreading,
        columns=df_with_spreading.columns[1:]
    )
    norm_data_path = os.path.join(method_out_dir, "normalized_data.csv")
    normalized_df.to_csv(norm_data_path, index=False)
    print(f"  Saved normalized data (with spreading): {norm_data_path}")

    df_1 = normalized_df.copy()
    gra_spread_col = find_column(df_1, "Average Spreading Ability")
    if gra_spread_col is None:
        print("  [Error] 'Average Spreading Ability' not found in normalized data. Skipping GRA.")
        return

    Y0 = df_1[gra_spread_col].values
    Yj = df_1.drop(columns=[gra_spread_col]).values

    a = np.min(np.abs(Y0[:, np.newaxis] - Yj))
    b = np.max(np.abs(Y0[:, np.newaxis] - Yj))
    rho = 0.5

    R = (a + rho * b) / (np.abs(Y0[:, np.newaxis] - Yj) + rho * b)
    grey_relational_degree = np.mean(R, axis=0)

    weights = np.array(grey_relational_degree) / np.sum(grey_relational_degree)
    print("  [D_Topsis] Weights:", weights)

    weighted_matrix = weights * normalized_matrix_without_spreading
    weighted_matrix_df = pd.DataFrame(weighted_matrix)
    weighted_matrix_path = os.path.join(method_out_dir, "weighted_matrix.csv")
    weighted_matrix_df.to_csv(weighted_matrix_path, index=False, header=False)
    print(f"  [D_Topsis] Saved weighted_matrix: {weighted_matrix_path}")

    positive_ideal_object = np.max(weighted_matrix, axis=0)
    negative_ideal_object = np.min(weighted_matrix, axis=0)

    euclidean_distance_positive = np.sqrt(np.sum((weighted_matrix - positive_ideal_object) ** 2, axis=1))
    euclidean_distance_negative = np.sqrt(np.sum((weighted_matrix - negative_ideal_object) ** 2, axis=1))

    relative_closeness = euclidean_distance_negative / (
        euclidean_distance_positive + euclidean_distance_negative
    )

    nodes_series = df_with_spreading[node_col_attr]
    node_closeness = list(zip(nodes_series, relative_closeness))

    df_node_closeness = pd.DataFrame(node_closeness, columns=["Node", "Relative Closeness"])
    node_influence_path = os.path.join(method_out_dir, "node_influence.csv")
    df_node_closeness.to_csv(node_influence_path, index=False)
    print(f"  [D_Topsis] Node influence saved: {node_influence_path}")

    avg_for_rank, infl_sorted = jaccard_and_rank_common(
        df_spread,
        spread_val_col,
        spread_node_col,
        df_node_closeness,
        method_out_dir,
        label="D_Topsis",
    )

    auto_kendall_for_lambdas(
        network_dir=network_dir,
        df_node_closeness=df_node_closeness,
        df_spread=df_spread,
        spread_val_col=spread_val_col,
        method_out_dir=method_out_dir,
    )


# ==========================
# METHOD 4: WSM
# ==========================

def run_wsm_method_for_network(network_name: str):
    print(f"\n=== [Method 4 - WSM] Network: {network_name} ===")

    network_dir = os.path.join(BASE_OUTPUTS_DIR, network_name)
    if not os.path.isdir(network_dir):
        print(f"  [Skip] {network_dir} is not a folder.")
        return

    attr_file = find_attribute_file(network_name)
    if attr_file is None:
        print("  [Skip] No attribute CSV found matching pattern for this network.")
        return

    baseline_spread_file = find_baseline_spreading_file(network_dir)
    if baseline_spread_file is None:
        print("  [Skip] No baseline spreading CSV found for WSM method.")
        return

    print(f"  Using attributes: {attr_file}")
    print(f"  Using baseline spreading:  {baseline_spread_file}")

    method_out_dir = os.path.join(network_dir, METHOD4_SUBFOLDER)
    os.makedirs(method_out_dir, exist_ok=True)

    df_attr = pd.read_csv(attr_file)
    df_spread = pd.read_csv(baseline_spread_file)

    node_col_attr = find_column(df_attr, "Node")
    if node_col_attr is None:
        print("  [Error] 'Node' column not found in attributes. Skipping.")
        return

    real_cols = {}
    for col in D_TOPSIS_COLUMNS:
        c = find_column(df_attr, col)
        if c is None:
            print(f"  [Error] WSM centrality '{col}' not found in attributes. Skipping method for this network.")
            return
        real_cols[col] = c

    spread_val_col = find_column(df_spread, "Average Spreading Ability")
    spread_node_col = find_column(df_spread, "Starting Node") or find_column(df_spread, "Node")
    if spread_val_col is None or spread_node_col is None:
        print("  [Error] Spreading file does not have proper node/value columns. Skipping.")
        return

    selected_cols = [node_col_attr] + [real_cols[c] for c in D_TOPSIS_COLUMNS]
    df_attr_sel = df_attr[selected_cols].copy()

    df_spread_ren = df_spread[[spread_node_col, spread_val_col]].copy()
    df_spread_ren.rename(
        columns={
            spread_node_col: node_col_attr,
            spread_val_col: "Average Spreading Ability",
        },
        inplace=True,
    )

    final_df = pd.merge(
        df_attr_sel,
        df_spread_ren,
        on=node_col_attr,
        how="inner"
    )

    if final_df.empty:
        print("  [Error] No common nodes between attributes and spreading. Skipping.")
        return

    if len(final_df) < len(df_attr_sel):
        print(f"  [Info] {len(df_attr_sel) - len(final_df)} nodes dropped due to no spreading info.")

    df_with_spreading = final_df.copy()
    cent_cols_only = [node_col_attr] + [real_cols[c] for c in D_TOPSIS_COLUMNS]
    df_without_spreading = final_df[cent_cols_only].copy()

    out_without_spread_xlsx = os.path.join(method_out_dir, "centrality_results_without_spreading.xlsx")
    out_with_spread_xlsx = os.path.join(method_out_dir, "centrality_results_with_spreading.xlsx")
    df_without_spreading.to_excel(out_without_spread_xlsx, index=False)
    df_with_spreading.to_excel(out_with_spread_xlsx, index=False)
    print(f"  [WSM] Saved Excel: {out_without_spread_xlsx}, {out_with_spread_xlsx}")

    matrix_without_spreading = df_without_spreading.drop(columns=[node_col_attr]).values.copy()
    matrix_with_spreading = df_with_spreading.drop(columns=[node_col_attr]).values

    col_sum = matrix_without_spreading.sum(axis=0)
    col_sum[col_sum == 0] = 1.0
    normalized_matrix_without_spreading = matrix_without_spreading / col_sum

    norm_without_path = os.path.join(method_out_dir, "normalized_centrality_matrix_without_spreading.csv")
    np.savetxt(norm_without_path, normalized_matrix_without_spreading, delimiter=",")
    print(f"  [WSM] Saved normalized matrix (without spreading): {norm_without_path}")

    col_min = matrix_with_spreading.min(axis=0)
    col_sum2 = matrix_with_spreading.sum(axis=0)
    col_sum2[col_sum2 == 0] = 1.0
    normalized_matrix_with_spreading = (matrix_with_spreading - col_min) / col_sum2

    normalized_df = pd.DataFrame(
        normalized_matrix_with_spreading,
        columns=df_with_spreading.columns[1:]
    )
    norm_data_path = os.path.join(method_out_dir, "normalized_data.csv")
    normalized_df.to_csv(norm_data_path, index=False)
    print(f"  [WSM] Saved normalized data (with spreading): {norm_data_path}")

    df_1 = normalized_df.copy()
    gra_spread_col = find_column(df_1, "Average Spreading Ability")
    if gra_spread_col is None:
        print("  [Error] 'Average Spreading Ability' not found in normalized data. Skipping GRA.")
        return

    Y0 = df_1[gra_spread_col].values
    Yj = df_1.drop(columns=[gra_spread_col]).values

    a = np.min(np.abs(Y0[:, np.newaxis] - Yj))
    b = np.max(np.abs(Y0[:, np.newaxis] - Yj))
    rho = 0.5

    R = (a + rho * b) / (np.abs(Y0[:, np.newaxis] - Yj) + rho * b)
    grey_relational_degree = np.mean(R, axis=0)

    weights = np.array(grey_relational_degree) / np.sum(grey_relational_degree)
    print("  [WSM] Weights:", weights)

    weighted_matrix = weights * normalized_matrix_without_spreading
    weighted_matrix_df = pd.DataFrame(weighted_matrix)
    weighted_matrix_path = os.path.join(method_out_dir, "weighted_matrix.csv")
    weighted_matrix_df.to_csv(weighted_matrix_path, index=False, header=False)
    print(f"  [WSM] Saved weighted_matrix: {weighted_matrix_path}")

    weighted_sums = np.sum(weighted_matrix, axis=1)
    node_series = df_with_spreading[node_col_attr]

    df_wsm = pd.DataFrame({"Node": node_series, "Weighted_Sum": weighted_sums})
    weighted_sums_path = os.path.join(method_out_dir, "weighted_sums.csv")
    df_wsm.to_csv(weighted_sums_path, index=False)
    print(f"  [WSM] Weighted sums saved: {weighted_sums_path}")

    df_node_closeness = pd.DataFrame({"Node": node_series, "Relative Closeness": weighted_sums})
    node_influence_path = os.path.join(method_out_dir, "node_influence.csv")
    df_node_closeness.to_csv(node_influence_path, index=False)
    print(f"  [WSM] Node influence saved: {node_influence_path}")

    avg_for_rank, infl_sorted = jaccard_and_rank_common(
        df_spread,
        spread_val_col,
        spread_node_col,
        df_node_closeness,
        method_out_dir,
        label="WSM",
    )

    infl_rank_path = os.path.join(method_out_dir, "node_influence_ranking.csv")
    wsm_rank_path = os.path.join(method_out_dir, "node_influence_ranking_wsm.csv")
    if os.path.isfile(infl_rank_path):
        pd.read_csv(infl_rank_path).to_csv(wsm_rank_path, index=False)
        print(f"  [WSM] Saved {wsm_rank_path}")

    auto_kendall_for_lambdas(
        network_dir=network_dir,
        df_node_closeness=df_node_closeness,
        df_spread=df_spread,
        spread_val_col=spread_val_col,
        method_out_dir=method_out_dir,
    )


# ==========================
# MASTER LOOP
# ==========================

def main():
    print("Base outputs dir:", BASE_OUTPUTS_DIR)

    if not os.path.isdir(BASE_OUTPUTS_DIR):
        print("ERROR: BASE_OUTPUTS_DIR does not exist.")
        return

    for entry in os.listdir(BASE_OUTPUTS_DIR):
        network_dir = os.path.join(BASE_OUTPUTS_DIR, entry)
        if not os.path.isdir(network_dir):
            continue

        network_name = entry

        # Method 1: centrality vs spreading + centrality vs lambda Kendall
        run_jaccard_centralities_for_network(network_name)

        # Method 2: Random attributes + GRA + TOPSIS + Jaccard + Kendall
        run_random_method_for_network(network_name)

        # Method 3: D TOPSIS method
        run_d_topsis_method_for_network(network_name)

        # Method 4: WSM method
        run_wsm_method_for_network(network_name)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
