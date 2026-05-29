# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 21:57:45 2025

@author: Javaria
"""

# ===========================
# Reproducibility + Threads
# ===========================
import os, random, numpy as np
os.environ["OMP_NUM_THREADS"]  = "1"
os.environ["MKL_NUM_THREADS"]  = "1"
os.environ["PYTHONHASHSEED"]   = "0"

# ===========================
# Core imports
# ===========================
import re
import glob
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

# ========= CONFIG =========
INPUT_DIR   = r"....\feature\Attribute_codes"     # .gr + graph_attributes_*_check.csv
#OUTPUT_ROOT = os.path.join(INPUT_DIR, "outputs_final_last")                          # stage-1 outputs live here
OUTPUT_ROOT = os.path.join(INPUT_DIR, "outputs") 
METHOD_TO_SELECTED_XLSX = {
    "laplacian": ["selected_features_lap.xlsx"],
    "mcfs":      ["selected_features_mcfs.xlsx"],
    "pca":       ["selected_features_pca.xlsx"],
    "spec":      ["selected_features_spec.xlsx"],
    "udfs":      ["selected_features_udfsBB.xlsx", "selected_features_udfs.xlsx"],
}
METHOD_TO_INFLUENCE_CSV = {
    "laplacian": "node_influence_lap_1.csv",
    "mcfs":      "node_influence_mcfs_1.csv",
    "pca":       "node_influence_pca_1.csv",
    "spec":      "node_influence_spec_1.csv",
    "udfs":      "node_influence_udfs_1.csv",
}

SIR_MAX_TIME_STEPS   = 100
SIR_NUM_SIMULATIONS  = 100
SIR_GAMMA            = 1.0
# =========================


# ---------- .gr reader preserving first-seen node order ----------
def read_gr_file_preserve_order(filename):
    G = nx.Graph()
    node_order = []
    seen = set()
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('a'):
                _, node1, node2 = line.split()
                node1 = int(node1)
                node2 = int(node2)
                G.add_edge(node1, node2)
                for node in (node1, node2):
                    if node not in seen:
                        seen.add(node)
                        node_order.append(node)
    return G, node_order


# ---------- lambda per-graph: beta_c = <k> / (<k^2> - <k>) ----------
def calculate_beta_c_from_graph(G):
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    N = len(degrees)
    if N == 0:
        return 0.0
    k_avg = degrees.mean()
    k2_avg = np.mean(degrees**2)
    denom = (k2_avg - k_avg)
    if denom <= 0:
        return 1e-12
    return k_avg / denom


# ---------- SIR simulator (per-start-node seed = i+4) ----------
def simulate_sir_model(G, lambda_value, gamma=SIR_GAMMA,
                       max_time_steps=SIR_MAX_TIME_STEPS, num_simulations=SIR_NUM_SIMULATIONS):
    starting_nodes = sorted(G.nodes())
    average_spreading_abilities = []

    for i, initial_infected_node in enumerate(starting_nodes):
        np.random.seed(i + 4)  # deterministic per start node

        cumulative_spreading_at_each_time_step = []
        for _ in range(num_simulations):
            infected_nodes = set([initial_infected_node])
            recovered_nodes = set()
            cumulative_spreading_simulations = []

            for _t in range(max_time_steps):
                newly_infected = set()
                newly_recovered = set()

                for node in infected_nodes:
                    for neighbor in sorted(G.neighbors(node)):
                        if neighbor not in infected_nodes and neighbor not in recovered_nodes:
                            if np.random.random() < lambda_value:
                                newly_infected.add(neighbor)

                for infected_node in list(sorted(infected_nodes)):
                    if np.random.random() < gamma:
                        newly_recovered.add(infected_node)
                        infected_nodes.remove(infected_node)

                infected_nodes.update(newly_infected)
                recovered_nodes.update(newly_recovered)

                cumulative_spreading_simulations.append(len(infected_nodes) + len(recovered_nodes))

            cumulative_spreading_at_each_time_step.append(cumulative_spreading_simulations)

        avg_curve = np.mean(cumulative_spreading_at_each_time_step, axis=0)
        average_spreading_abilities.append(avg_curve)

    return average_spreading_abilities, starting_nodes


# ---------- helpers ----------
def parse_dataset_from_attrs(path):
    base = os.path.basename(path)
    m = re.match(r"graph_attributes_(.+?)_check\.csv$", base)
    if not m:
        raise ValueError(f"Bad filename: {base}")
    return m.group(1)

_STEM_RE = re.compile(r"^(.*?)(?:-(?:[gc])?\d+)+$")   # captures stem before trailing -3 / -g2455 / -c1 etc.
_LAST_NUM_RE = re.compile(r"-(?:[gc])?(\d+)$")        # captures last trailing number

def stem_of(name_no_ext: str) -> str:
    m = _STEM_RE.match(name_no_ext)
    return m.group(1) if m else name_no_ext

def last_suffix_num(name_no_ext: str):
    m = _LAST_NUM_RE.search(name_no_ext)
    return int(m.group(1)) if m else None

def find_selected_file(method_dir, candidates):
    for name in candidates:
        p = os.path.join(method_dir, name)
        if os.path.isfile(p):
            return p
    return None


# ---------- flexible dataset-folder matching ----------
def _norm_name(x: str) -> str:
    """
    Normalize name for comparison:
    - lowercase
    - replace '_' with '-'
    - take STEM (before -gNN / -cNN etc.)
    """
    x_no_ext = os.path.splitext(x)[0]
    x_repl   = x_no_ext.replace("_", "-").lower()
    return stem_of(x_repl)

def find_dataset_folder_for_attrs(attrs_dataset, all_folders):
    """
    attrs_dataset: name parsed from attributes file (e.g. 'ENZYMES-g296-c1')
    all_folders: list of folder names under OUTPUT_ROOT (e.g. ['ENZYMES_g296-c1', ...])

    Tries:
      1) exact match
      2) '-' <-> '_' variants
      3) stem + numeric suffix matching (similar to .gr logic)
      4) loose normalized substring
    """
    # 1) exact folder name match
    if attrs_dataset in all_folders:
        return attrs_dataset

    # 2) direct dash/underscore swap variants
    alt1 = attrs_dataset.replace("-", "_")
    alt2 = attrs_dataset.replace("_", "-")
    for cand in (alt1, alt2):
        if cand in all_folders:
            return cand

    # 3) stem-based & numeric-suffix based
    target_stem = _norm_name(attrs_dataset)
    target_num  = last_suffix_num(attrs_dataset)
    candidates  = []
    for folder in all_folders:
        s = _norm_name(folder)
        if s == target_stem:
            n = last_suffix_num(folder)
            candidates.append((folder, n))

    if candidates:
        if target_num is not None:
            # pick folder with closest numeric suffix
            candidates.sort(key=lambda t: (abs((t[1] or 10**9) - target_num), t[1] is None))
        else:
            candidates.sort(key=lambda t: (t[1] is None, t[1] if t[1] is not None else 10**9))
        return candidates[0][0]

    # 4) loose fallback: normalized substring (remove -/_ and lowercase)
    raw_norm = attrs_dataset.replace("-", "").replace("_", "").lower()
    for folder in all_folders:
        if raw_norm in folder.replace("-", "").replace("_", "").lower():
            return folder

    # nothing found
    return None


def find_gr_for_dataset(dataset_name):
    """
    Match .gr by dataset name:
      1) exact '<dataset>.gr'
      2) same STEM match: e.g., 'aves-weaver-social-3' -> matches 'aves-weaver-social-16.gr'
         - if multiple, choose closest numeric suffix to dataset's suffix
      3) fallback: any '*<dataset>*\.gr' (rare)
    """
    # 1) exact
    exact = os.path.join(INPUT_DIR, f"{dataset_name}.gr")
    if os.path.isfile(exact):
        return exact

    # 2) stem-based
    ds_stem = stem_of(dataset_name)
    ds_num  = last_suffix_num(dataset_name)

    gr_candidates = glob.glob(os.path.join(INPUT_DIR, "*.gr"))
    same_stem = []
    for p in gr_candidates:
        name_no_ext = os.path.splitext(os.path.basename(p))[0]
        if stem_of(name_no_ext) == ds_stem:
            num = last_suffix_num(name_no_ext)
            same_stem.append((p, num))

    if same_stem:
        if ds_num is not None:
            # pick closest numeric suffix
            same_stem.sort(key=lambda t: (abs((t[1] or 10**9) - ds_num), t[1] is None))
            return same_stem[0][0]
        # else: no numeric in dataset; just pick first (sorted for stability)
        same_stem.sort(key=lambda t: (t[1] is None, t[1] if t[1] is not None else 10**9))
        return same_stem[0][0]

    # 3) loose fallback
    loose = glob.glob(os.path.join(INPUT_DIR, f"*{dataset_name}*.gr"))
    return loose[0] if loose else None


# ---------- main per-dataset/method pipeline ----------
def process_dataset_method(attrs_dataset, folder_dataset, method):
    """
    attrs_dataset: name used in attributes/gr files (e.g. 'ENZYMES-g296-c1')
    folder_dataset: actual folder name under outputs/ (e.g. 'ENZYMES_g296-c1')
    method: 'laplacian', 'mcfs', ...
    """
    attrs_csv  = os.path.join(INPUT_DIR,  f"graph_attributes_{attrs_dataset}_check.csv")
    gr_path    = find_gr_for_dataset(attrs_dataset)
    method_dir = os.path.join(OUTPUT_ROOT, folder_dataset, method)

    if not os.path.isdir(method_dir):
        print(f"  [skip] {attrs_dataset}/{method}: method dir missing ({method_dir})")
        return
    if not os.path.isfile(attrs_csv):
        print(f"  [skip] {attrs_dataset}/{method}: attributes CSV missing ({attrs_csv})")
        return
    if not gr_path or not os.path.isfile(gr_path):
        print(f"  [skip] {attrs_dataset}/{method}: GR file missing (looked in {INPUT_DIR})")
        return

    # --- 1) lambda from graph + SIR -> average_spreading_abilities.csv ---
    G, node_order = read_gr_file_preserve_order(gr_path)
    lambda_value = calculate_beta_c_from_graph(G)

    sir_csv = os.path.join(method_dir, "average_spreading_abilities.csv")
    if not os.path.isfile(sir_csv):
        avg_spread_all, sorted_nodes = simulate_sir_model(
            G,
            lambda_value=lambda_value,
            gamma=SIR_GAMMA,
            max_time_steps=SIR_MAX_TIME_STEPS,
            num_simulations=SIR_NUM_SIMULATIONS
        )
        # IMPORTANT: final-time-step values only
        node_to_result = {node: avg[-1] for node, avg in zip(sorted_nodes, avg_spread_all)}
        ordered_results = [node_to_result[node] for node in node_order]

        pd.DataFrame({
            'Starting Node': node_order,
            'Average Spreading Ability': ordered_results
        }).to_csv(sir_csv, index=False)

        pd.DataFrame([{"dataset": attrs_dataset, "method": method, "lambda_beta_c": float(lambda_value)}]) \
          .to_csv(os.path.join(method_dir, "sir_params.csv"), index=False)
        print(f"  ✓ {attrs_dataset}/{method}: average_spreading_abilities.csv (lambda=βc={lambda_value:.6g})")
    else:
        print(f"  = {attrs_dataset}/{method}: average_spreading_abilities.csv exists (keeping)")

    # --- 2) selected features for this method ---
    sel_path = find_selected_file(method_dir, METHOD_TO_SELECTED_XLSX.get(method, []))
    if not sel_path:
        print(f"  [skip] {attrs_dataset}/{method}: selected features file missing in {method_dir}")
        return

    dolphin_data = pd.read_csv(attrs_csv)
    feature_data = pd.read_excel(sel_path)
    average_spreading_data = pd.read_csv(sir_csv)

    # --- 3) selected_data_1 ---
    selected_columns = ['Node'] + [col for col in feature_data.columns if col in dolphin_data.columns]
    selected_data_1 = dolphin_data[selected_columns].copy()
    selected_data_1['Average Spreading Ability'] = average_spreading_data['Average Spreading Ability']
    selected_data_1.to_excel(os.path.join(method_dir, "selected_data_output.xlsx"), index=False)

    # --- 4) MinMax normalize (exclude 'Node') ---
    scaler = MinMaxScaler()
    selected_data = selected_data_1.copy()
    cols_to_norm = [c for c in selected_data.columns if c != 'Node']
    selected_data[cols_to_norm] = scaler.fit_transform(selected_data[cols_to_norm])

    # --- 5) Save centrality + matrices ---
    df_wo = selected_data.drop(columns=['Average Spreading Ability'])
    df_wo.to_excel(os.path.join(method_dir, 'centrality_results_without_spreading.xlsx'), index=False)
    mat_wo = df_wo.drop(columns=['Node']).values.copy()
    pd.DataFrame(mat_wo).to_csv(os.path.join(method_dir, 'matrix_without_spreading.csv'), index=False, header=None)

    df_w  = selected_data
    df_w.to_excel(os.path.join(method_dir, 'centrality_results_with_spreading.xlsx'), index=False)
    mat_w = df_w.drop(columns=['Node']).values
    pd.DataFrame(mat_w).to_csv(os.path.join(method_dir, 'matrix_with_spreading.csv'), index=False, header=None)

    np.savetxt(os.path.join(method_dir, 'normalized_centrality_matrix_without_spreading.csv'), mat_wo, delimiter=',')
    norm_df = pd.DataFrame(mat_w, columns=df_w.columns[1:])  # exclude Node
    norm_df.to_csv(os.path.join(method_dir, 'normalized_data.csv'), index=False)

    # --- 6) GRA + TOPSIS ---
    df_1 = norm_df
    if 'Average Spreading Ability' not in df_1.columns:
        print(f"  [skip] {attrs_dataset}/{method}: 'Average Spreading Ability' missing post-normalization")
        return

    Y0 = df_1['Average Spreading Ability'].values
    Yj = df_1.drop(columns=['Average Spreading Ability']).values

    a = np.min(np.abs(Y0[:, np.newaxis] - Yj))
    b = np.max(np.abs(Y0[:, np.newaxis] - Yj))
    rho = 0.5
    R = (a + rho * b) / (np.abs(Y0[:, np.newaxis] - Yj) + rho * b)
    grey_relational_degree = np.mean(R, axis=0)
    weights = np.array(grey_relational_degree) / np.sum(grey_relational_degree)

    weighted_matrix = weights * mat_wo
    pos_ideal = np.max(weighted_matrix, axis=0)
    neg_ideal = np.min(weighted_matrix, axis=0)
    d_pos = np.sqrt(np.sum((weighted_matrix - pos_ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weighted_matrix - neg_ideal) ** 2, axis=1))
    relative_closeness = d_neg / (d_pos + d_neg)

    nodes = dolphin_data['Node']
    node_closeness = list(zip(nodes, relative_closeness))

    infl_csv = os.path.join(method_dir, METHOD_TO_INFLUENCE_CSV.get(method, f"node_influence_{method}_1.csv"))
    pd.DataFrame(node_closeness, columns=['Node', 'Relative Closeness']).to_csv(infl_csv, index=False)
    print(f"  ✓ {attrs_dataset}/{method}: {os.path.basename(infl_csv)}")


# ---------- driver ----------
def main():
    attr_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "graph_attributes_*_check.csv")))
    if not attr_paths:
        print("No attributes files found in:", INPUT_DIR)
        return

    # all dataset folders actually present under outputs
    all_dataset_folders = [
        d for d in os.listdir(OUTPUT_ROOT)
        if os.path.isdir(os.path.join(OUTPUT_ROOT, d))
    ]

    for attrs_csv in attr_paths:
        attrs_dataset = parse_dataset_from_attrs(attrs_csv)

        # find best matching folder name under outputs for this attrs_dataset
        folder_dataset = find_dataset_folder_for_attrs(attrs_dataset, all_dataset_folders)
        if folder_dataset is None:
            print(f"\n[warn] No matching outputs folder found for attributes dataset '{attrs_dataset}'")
            continue

        dataset_root = os.path.join(OUTPUT_ROOT, folder_dataset)

        method_dirs = [
            d for d in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, d))
        ]
        methods = [m for m in method_dirs if m in METHOD_TO_SELECTED_XLSX]

        print(f"\n=== Dataset (attrs): {attrs_dataset}  |  outputs folder: {folder_dataset} ===")
        if not methods:
            print("  [warn] No recognized method subfolders here.")
            continue

        for m in methods:
            print(f" - Method: {m}")
            process_dataset_method(attrs_dataset, folder_dataset, m)


if __name__ == "__main__":
    main()
