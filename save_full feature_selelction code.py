# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 21:34:48 2025

@author: Javaria
"""







import os, random, numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

random.seed(42)
np.random.seed(42)

# ===========================
# Core Imports
# ===========================
from scipy.sparse import *
from scipy.sparse import diags
from sklearn.preprocessing import StandardScaler
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score
from skfeature.function.sparse_learning_based import MCFS, UDFS
from skfeature.utility.sparse_learning import feature_ranking as udfs_feature_ranking
from skfeature.function.similarity_based import SPEC
from sklearn.decomposition import PCA
import pandas as pd
import glob, re








import os
import re
import glob
import numpy as np
import pandas as pd

# ----- EXACT-LIKE imports per your snippets -----
from scipy.sparse import *
from scipy.sparse import diags
from sklearn.preprocessing import StandardScaler

# Laplacian
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score

# MCFS
from skfeature.function.sparse_learning_based import MCFS

# SPEC
from skfeature.function.similarity_based import SPEC

# UDFS
from skfeature.function.sparse_learning_based import UDFS
from skfeature.utility.sparse_learning import feature_ranking as udfs_feature_ranking

# PCA
from sklearn.decomposition import PCA
# ------------------------------------------------
#put code file, atrributes and network.gr files in same folder
# ========= CONFIG =========
INPUT_DIR = r"./"                      # folder with your CSVs
FILE_GLOB = "graph_attributes_*_check.csv"
OUTPUT_ROOT = r"./outputs_final_last"
NUM_FEATURES = 25
# =========================
def reset_seed(seed=42):
    import os, random, numpy as np
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def parse_dataset_name(path):
    """
    graph_attributes_<DATASET>_check.csv -> <DATASET>
    """
    base = os.path.basename(path)
    m = re.match(r"graph_attributes_(.+?)_check\.csv$", base)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {base}")
    return m.group(1)

# ---------- Your helper kept 1:1 for Laplacian-style loaders ----------
def load_data_from_csv(file_path):
    """
    Load data from a CSV file.
    Returns:
      Y (numpy array) -> data excluding first column
      column_names (Index) -> names excluding first column
    """
    data = pd.read_csv(file_path)
    Y = data.iloc[:, 1:].values  # Exclude the first column
    column_names = data.columns[1:]  # Get column names excluding the first column
    return Y, column_names

def feature_ranking_laplacian(score):
    """
    EXACT same ranking logic for Laplacian.
    """
    idx = np.argsort(score, 0)
    return idx
# ---------------------------------------------------------------------

# -------------------- METHOD RUNNERS (keep your semantics) --------------------
def run_laplacian(Y, column_names, out_dir):
    # Standardize (same as your code)
    scaler = StandardScaler()
    X = scaler.fit_transform(Y)

    # construct affinity matrix (same kwargs you used)
    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn",
                "weight_mode": "heat_kernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X, **kwargs_W)

    # scores and ranking (same)
    score = lap_score.lap_score(X, W=W)
    idx = feature_ranking_laplacian(score)

    # select top NUM_FEATURES
    selected_features = X[:, idx[0:NUM_FEATURES]]
    selected_lap_df = pd.DataFrame(selected_features,
                                   columns=column_names[idx[0:NUM_FEATURES]])

    safe_mkdir(out_dir)
    excel_file_path = os.path.join(out_dir, 'selected_features_lap.xlsx')
    selected_lap_df.to_excel(excel_file_path, index=False)

    # also optional: full ranking table (helpful in next steps)
    ranking_csv = os.path.join(out_dir, 'feature_ranking_laplacian.csv')
    pd.DataFrame({
        "feature": np.array(column_names)[idx.flatten()],
        "score": np.array(score)[idx.flatten()],
        "rank": np.arange(1, len(idx)+1)
    }).to_csv(ranking_csv, index=False)

def run_mcfs(Y, column_names, out_dir):
    reset_seed(42)   # 🔴 ADD THIS
    scaler = StandardScaler()
    X = scaler.fit_transform(Y)


    kwargs = {"metric": "euclidean", "neighborMode": "knn",
              "weightMode": "heatKernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X, **kwargs)

    num_fea = NUM_FEATURES
    #  code used n_clusters=10
    Weight = MCFS.mcfs(X, n_selected_features=num_fea, W=W, n_clusters=10)
    idx = MCFS.feature_ranking(Weight)

    selected_features = X[:, idx[0:num_fea]]
    selected_mcfs_df = pd.DataFrame(selected_features,
                                    columns=column_names[idx[0:num_fea]])

    safe_mkdir(out_dir)
    excel_file_path = os.path.join(out_dir, 'selected_features_mcfs.xlsx')
    selected_mcfs_df.to_excel(excel_file_path, index=False)

    # save ranking weights too (handy)
    ranking_csv = os.path.join(out_dir, 'feature_ranking_mcfs.csv')
    pd.DataFrame({
        "feature": np.array(column_names)[idx],
        "rank": np.arange(1, len(idx)+1)
    }).to_csv(ranking_csv, index=False)

def run_pca(Y, column_names, out_dir):
    # replicate your logic
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Y)

    model = PCA(n_components=5)
    results = model.fit_transform(X_scaled)

    loadings = model.components_.T  # features x components
    feature_loadings = np.abs(loadings).sum(axis=1)

    # top 25 by cumulative abs loadings (same as your code)
    top_features_indices = feature_loadings.argsort()[-NUM_FEATURES:][::-1]
    top_features = np.array(column_names)[top_features_indices]

    # Save the "top_features_loadings.xlsx" like your code
    safe_mkdir(out_dir)
    top_features_df = pd.DataFrame(
        pd.DataFrame(X_scaled, columns=column_names)[top_features]
    )
    excel_top = os.path.join(out_dir, 'top_features_loadings.xlsx')
    top_features_df.to_excel(excel_top, index=False)

    # Save the selected features values (same filename as your code)
    selected_features_df = pd.DataFrame(X_scaled[:, top_features_indices],
                                        columns=top_features)
    excel_sel = os.path.join(out_dir, 'selected_features_pca.xlsx')
    selected_features_df.to_excel(excel_sel, index=False)

    # optional: also save explained variance and raw loadings
    pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(model.explained_variance_ratio_))],
        "explained_variance_ratio": model.explained_variance_ratio_
    }).to_csv(os.path.join(out_dir, "pca_variance.csv"), index=False)

    loadings_df = pd.DataFrame(loadings, columns=[f"PC{i+1}" for i in range(loadings.shape[1])])
    loadings_df.insert(0, "feature", column_names.values)
    loadings_df.to_csv(os.path.join(out_dir, "pca_loadings.csv"), index=False)

def run_spec(Y, column_names, out_dir):
    scaler = StandardScaler()
    X = scaler.fit_transform(Y)

    # your style=0
    kwargs = {'style': 0}
    score = SPEC.spec(X, **kwargs)
    idx = SPEC.feature_ranking(score, **kwargs)  # SPEC sorts descending internally for style 0

    num_fea = NUM_FEATURES
    selected_features = X[:, idx[0:num_fea]]
    selected_spec_df = pd.DataFrame(selected_features,
                                    columns=column_names[idx[0:num_fea]])

    safe_mkdir(out_dir)
    excel_file_path = os.path.join(out_dir, 'selected_features_spec.xlsx')
    selected_spec_df.to_excel(excel_file_path, index=False)

    # save ranking table
    ranking_csv = os.path.join(out_dir, 'feature_ranking_spec.csv')
    pd.DataFrame({
        "feature": np.array(column_names)[idx],
        "score": np.array(score)[idx],
        "rank": np.arange(1, len(idx)+1)
    }).to_csv(ranking_csv, index=False)

def run_udfs(Y, column_names, out_dir):
    reset_seed(42)   # 🔴 ADD THIS
    scaler = StandardScaler()
    X = scaler.fit_transform(Y)

    num_cluster = 10  # as in your code
    Weight = UDFS.udfs(X, gamma=0.1, n_clusters=num_cluster)
    idx = udfs_feature_ranking(Weight)

    selected_features = X[:, idx[0:NUM_FEATURES]]
    selected_udfs_df = pd.DataFrame(selected_features,
                                    columns=column_names[idx[0:NUM_FEATURES]])

    safe_mkdir(out_dir)
    # keep your original filename 'selected_features_udfsBB.xlsx'
    excel_file_path = os.path.join(out_dir, 'selected_features_udfsBB.xlsx')
    selected_udfs_df.to_excel(excel_file_path, index=False)

    # and store raw weights/ranking for reference
    pd.DataFrame({
        "feature": np.array(column_names)[idx],
        "rank": np.arange(1, len(idx)+1)
    }).to_csv(os.path.join(out_dir, "feature_ranking_udfs.csv"), index=False)
# ------------------------------------------------------------------------------

METHODS = [
    ("laplacian", run_laplacian),
    ("mcfs",      run_mcfs),
    ("pca",       run_pca),
    ("spec",      run_spec),
    ("udfs",      run_udfs),
]

def main():
    csv_paths = sorted(glob.glob(os.path.join(INPUT_DIR, FILE_GLOB)))
    if not csv_paths:
        print(f"No files matched: {os.path.join(INPUT_DIR, FILE_GLOB)}")
        return

    print(f"Found {len(csv_paths)} files.")
    for file_path in csv_paths:
        try:
            dataset = parse_dataset_name(file_path)
            print(f"\n== Dataset: {dataset} ==")

            # load the raw matrix and names (your loader)
            Y, column_names = load_data_from_csv(file_path)

            # run all methods
            for tag, fn in METHODS:
                out_dir = os.path.join(OUTPUT_ROOT, dataset, tag)
                print(f" - Method: {tag}")
                fn(Y, column_names, out_dir)

        except Exception as e:
            print(f"[ERROR] while processing '{file_path}': {e}")

if __name__ == "__main__":
    main()
