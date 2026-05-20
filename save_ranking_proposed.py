

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 22:39:11 2025

@author: Javaria
"""

# ===========================
# Batch ranking + final merge
# (MODIFIED: SIR computed dynamically per rank)
# ===========================

import os
import glob
import math
import numpy as np
import pandas as pd
import networkx as nx


# --------- CONFIG ----------
#INPUT_DIR   = r"....\Attribute_codes"
#OUTPUT_ROOT = os.path.join(INPUT_DIR, "output_final_last")
INPUT_DIR   = r"....\Attribute_codes"
OUTPUT_ROOT = os.path.join(INPUT_DIR, "outputs_final_last")
METHODS = ["laplacian", "mcfs", "spec", "pca", "udfs"]

INFL_FILE_BY_METHOD = {
    "laplacian": "node_influence_lap_1.csv",
    "mcfs":      "node_influence_mcfs_1.csv",
    "spec":      "node_influence_spec_1.csv",
    "pca":       "node_influence_pca_1.csv",
    "udfs":      "node_influence_udfs_1.csv",
}

RANKED_FILENAMES = {
    "laplacian": "node_influence_ranking_lap.csv",
    "mcfs":      "node_influence_ranking_mcfs.csv",
    "spec":      "node_influence_ranking_spec.csv",
    "pca":       "node_influence_ranking_pca.csv",
    "udfs":      "node_influence_ranking_udfs.csv",
}

# SIR CONFIG
SIR_GAMMA = 1
SIR_MAX_TIME_STEPS = 100
SIR_NUM_SIMULATIONS = 100
# ---------------------------


# ============================================================
# Helper: read .gr file and preserve node order
# ============================================================

# ============================================================

# ============================================================
def read_gr_file_preserve_order(gr_path):
    """
    Reads .gr file, returns (Graph G, node_order list)
    Assumes .gr contains edges in "u v" format (standard DIMACS style).
    """
    G = nx.Graph()
    node_order = []

    with open(gr_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue

            parts = line.split()

            # DIMACS style: "p edge n m"
            if parts[0] == "p":
                continue

            # DIMACS style: "e u v"
            if parts[0] == "a" and len(parts) >= 3:
                u = int(parts[1])
                v = int(parts[2])

                if u not in G:
                    G.add_node(u)
                    node_order.append(u)
                if v not in G:
                    G.add_node(v)
                    node_order.append(v)

                G.add_edge(u, v)

            # If file is just "u v"
            elif len(parts) == 2:
                u = int(parts[0])
                v = int(parts[1])

                if u not in G:
                    G.add_node(u)
                    node_order.append(u)
                if v not in G:
                    G.add_node(v)
                    node_order.append(v)

                G.add_edge(u, v)

    return G, node_order


# ============================================================
# beta_c formula
# ============================================================
def calculate_beta_c_from_graph(G):
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    N = len(degrees)
    if N == 0:
        return 0.0

    k_avg = degrees.mean()
    k2_avg = np.mean(degrees ** 2)
    denom = (k2_avg - k_avg)

    if denom <= 0:
        return 1e-12

    return k_avg / denom


# ============================================================
# SIR simulation for one start node
# ============================================================
def simulate_sir_for_one_node(G, start_node, lambda_value,
                              gamma=SIR_GAMMA,
                              max_time_steps=SIR_MAX_TIME_STEPS,
                              num_simulations=SIR_NUM_SIMULATIONS,
                              seed_base=4):
    """
    Simulate SIR multiple times for one initial infected node.
    Returns final average spreading ability (last time step only).

    Seed matches your original code:
    np.random.seed(i + 4) where i is index in sorted(G.nodes()).
    """

    # EXACT SAME SEED LOGIC AS YOUR ORIGINAL CODE
    starting_nodes = sorted(G.nodes())
    node_to_index = {node: i for i, node in enumerate(starting_nodes)}

    if start_node not in node_to_index:
        return float("-inf")

    np.random.seed(node_to_index[start_node] + seed_base)

    cumulative_spreading_at_each_time_step = []

    for _ in range(num_simulations):
        infected_nodes = set([start_node])
        recovered_nodes = set()
        cumulative_spreading_simulation = []

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

            cumulative_spreading_simulation.append(len(infected_nodes) + len(recovered_nodes))

        cumulative_spreading_at_each_time_step.append(cumulative_spreading_simulation)

    avg_curve = np.mean(cumulative_spreading_at_each_time_step, axis=0)

    # IMPORTANT: final-time-step value only
    return float(avg_curve[-1])


# ============================================================
# Ranking utilities
# ============================================================
def dense_rank_desc(df, value_col, new_col="Rank"):
    out = df.copy()
    out[new_col] = out[value_col].rank(method="dense", ascending=False)
    return out


def make_rank_files_for_dataset(dataset_dir, dataset_name):
    """
    Read node_influence_*_1.csv from each method subfolder and rank -> save in dataset_dir
    Returns: list of ranked node_influence files created
    """
    ranked_files = []

    for m in METHODS:
        method_dir = os.path.join(dataset_dir, m)
        infl_in = os.path.join(method_dir, INFL_FILE_BY_METHOD[m])

        if not os.path.isfile(infl_in):
            continue

        try:
            df = pd.read_csv(infl_in)

            if "Relative Closeness" not in df.columns:
                raise ValueError(f"{infl_in} missing 'Relative Closeness' column")

            ranked = dense_rank_desc(
                df.sort_values(by="Relative Closeness", ascending=False),
                "Relative Closeness",
                "Rank"
            )

            ranked_path = os.path.join(dataset_dir, RANKED_FILENAMES[m])
            ranked.to_csv(ranked_path, index=False)

            ranked_files.append(ranked_path)

            print(f"  ✓ {dataset_name}: wrote {os.path.basename(ranked_path)}")

        except Exception as e:
            print(f"  [warn] {dataset_name}/{m}: failed to rank node influence: {e}")

    return ranked_files


# ============================================================
# Merge helpers
# ============================================================
def read_ranking_files(file_paths):
    """
    Build dict: {rank_number (int): set(nodes)} aggregated across all files.
    """
    all_ranked_nodes = {}

    for file_path in file_paths:
        print(f"\n>>> Reading ranking file: {file_path}")
        df = pd.read_csv(file_path, delimiter=',')

        # Detect node column safely
        df.columns = [c.strip() for c in df.columns]

        possible_node_cols = ["Node", "node", "Starting Node", "starting_node", "start_node"]
        node_col = next((c for c in possible_node_cols if c in df.columns), None)

        if node_col is None:
            raise ValueError(f"{file_path} has no Node column. Found: {df.columns.tolist()}")

        if "Rank" not in df.columns:
            raise ValueError(f"{file_path} missing Rank column")

        for row in df.itertuples(index=False):
            node_id = getattr(row, node_col)
            rank = int(getattr(row, "Rank"))
            all_ranked_nodes.setdefault(rank, set()).add(node_id)

    print("\n[DEBUG] Finished reading ranking files. Initial ranks content:")
    for r in sorted(all_ranked_nodes.keys()):
        print(f" Rank {r}: {sorted(all_ranked_nodes[r])}")

    return all_ranked_nodes


def remove_nodes_from_subsequent_ranks_preserve_empty(all_ranked_nodes):
    """
    Remove duplicates top-down while preserving ALL original rank keys.
    """
    print("\n>>> Removing duplicates across ranks (top-down)...")

    sorted_ranks = sorted(all_ranked_nodes.keys())

    for i, r in enumerate(sorted_ranks):
        current_nodes = all_ranked_nodes.get(r, set())

        print(f"\n Rank {r} BEFORE dedup: {sorted(current_nodes)}")

        if not current_nodes:
            continue

        for nr in sorted_ranks[i+1:]:
            if nr in all_ranked_nodes and all_ranked_nodes[nr]:
                before = len(all_ranked_nodes[nr])
                all_ranked_nodes[nr].difference_update(current_nodes)
                after = len(all_ranked_nodes[nr])

                if before != after:
                    print(f"   Removed {before - after} duplicates from rank {nr}")

    print("\n[DEBUG] After dedup, ranks content:")
    for r in sorted(all_ranked_nodes.keys()):
        print(f" Rank {r}: {sorted(all_ranked_nodes[r])}")

    return all_ranked_nodes


# ============================================================
# Final ranking using SIR dynamically per rank
# ============================================================
def create_final_ranking_using_sir_preserve_indices(all_ranked_nodes, G, lambda_value, output_file):
    ranks_sorted = sorted(all_ranked_nodes.keys())
    if not ranks_sorted:
        print("No ranks found; nothing to do.")
        return []

    sir_score_cache = {}

    min_rank = min(ranks_sorted)
    max_rank = max(ranks_sorted)

    for r in range(min_rank, max_rank + 1):
        all_ranked_nodes.setdefault(r, set())

    new_ranking_rows = []

    r = min_rank
    while r <= max_rank:
        nodes_here = all_ranked_nodes.get(r, set())

        print(f"\n>>> Processing Rank {r}")
        print(f" Current nodes: {sorted(nodes_here)}")

        if nodes_here:
            scores = {}

            for node in sorted(nodes_here):
                if node in sir_score_cache:
                    score = sir_score_cache[node]
                    print(f"  Node {node} reused cached spreading ability: {score}")
                else:
                    if node not in G:
                        print(f"  [Warning] Node {node} not in graph, skipping simulation (score = -inf).")
                        score = -math.inf
                    else:
                        score = simulate_sir_for_one_node(
                            G,
                            start_node=node,
                            lambda_value=lambda_value,
                            gamma=SIR_GAMMA,
                            max_time_steps=SIR_MAX_TIME_STEPS,
                            num_simulations=SIR_NUM_SIMULATIONS
                        )

                    sir_score_cache[node] = score
                    print(f"  Node {node} spreading ability computed: {score}")

                scores[node] = score

            max_score = max(scores.values()) if scores else -math.inf
            top_nodes = [n for n, sc in scores.items() if sc == max_score]

            print(f"  Top nodes at Rank {r}: {top_nodes} (ability = {max_score})")

            for n in top_nodes:
                new_ranking_rows.append((n, max_score, r))

            for n in top_nodes:
                nodes_here.discard(n)

            if nodes_here:
                print(f"  Remaining nodes to push from Rank {r} → Rank {r+1}: {sorted(nodes_here)}")

                next_rank = r + 1
                if next_rank > max_rank:
                    max_rank = next_rank

                all_ranked_nodes.setdefault(next_rank, set()).update(nodes_here)
                all_ranked_nodes[r] = set()

                print(
                    f"  After move: Rank {r} = {sorted(all_ranked_nodes[r])}, "
                    f"Rank {next_rank} = {sorted(all_ranked_nodes[next_rank])}"
                )
        else:
            print("  (empty rank)")

        r += 1

    df = pd.DataFrame(new_ranking_rows, columns=['Node', 'Spreading Ability', 'Rank'])
    df.to_csv(output_file, index=False)

    print(f"\n[DEBUG] Final ranking saved to '{output_file}'")

    return new_ranking_rows
def reorder_ranks_preserve_repeats(csv_path):
    """
    Remove gaps in Rank column but preserve repeated ranks and order.
    Example:
      [1,2,3,7,7,8,11,11] -> [1,2,3,4,4,5,6,6]
    """
    print("\n>>> Reordering final ranks (preserve repeats, remove gaps)")

    df = pd.read_csv(csv_path)

    old_ranks = df['Rank'].tolist()

    rank_map = {}
    new_ranks = []
    next_rank = 1

    for r in old_ranks:
        if r not in rank_map:
            rank_map[r] = next_rank
            next_rank += 1
        new_ranks.append(rank_map[r])

    df['Rank'] = new_ranks
    df.to_csv(csv_path, index=False)

    print("[DEBUG] Rank gaps removed, repeated ranks preserved.")


# ============================================================
# Main dataset processor
# ============================================================
def process_one_dataset(dataset_name):
    dataset_dir = os.path.join(OUTPUT_ROOT, dataset_name)
    if not os.path.isdir(dataset_dir):
        print(f"[skip] {dataset_name}: missing dataset folder under outputs")
        return

    print(f"\n=== Ranking for dataset: {dataset_name} ===")

    # Step-1: create ranked node influence files at dataset level
    ranked_files = make_rank_files_for_dataset(dataset_dir, dataset_name)

    if not ranked_files:
        print(f"  [skip] {dataset_name}: no node_influence files found to rank.")
        return

    # Step-2: locate .gr file
   # gr_candidates = glob.glob(os.path.join(INPUT_DIR, dataset_name, "*.gr"))
    #if not gr_candidates:
     #   print(f"  [skip] {dataset_name}: no .gr file found in {os.path.join(INPUT_DIR, dataset_name)}")
      #  return
    # Step-2: locate .gr file (search anywhere under INPUT_DIR)
    # Step-2: locate .gr file inside toy_example/<dataset>/<dataset>.gr
   # gr_path = os.path.join(INPUT_DIR, "toy_example", dataset_name, f"{dataset_name}.gr")
    gr_path = os.path.join(INPUT_DIR, f"{dataset_name}.gr")
    if not os.path.isfile(gr_path):
        print(f"  [skip] {dataset_name}: no .gr file found at {gr_path}")
        return

    print(f"  ✓ Using graph file: {gr_path}")

    # Step-3: read graph
    G, node_order = read_gr_file_preserve_order(gr_path)
    print(f"  ✓ Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Step-4: compute lambda_value (beta_c)
    lambda_value = calculate_beta_c_from_graph(G)
    print(f"  ✓ Computed lambda (beta_c) = {lambda_value}")

    # Step-5: final merge using SIR dynamically
    ranking_file_paths = []
    for m in METHODS:
        p = os.path.join(dataset_dir, RANKED_FILENAMES[m])
        if os.path.isfile(p):
            ranking_file_paths.append(p)

    if not ranking_file_paths:
        print(f"  [skip] {dataset_name}: no ranking files present.")
        return

    output_file = os.path.join(dataset_dir, f"final_ranking_{dataset_name}.csv")

    all_ranked_nodes = read_ranking_files(ranking_file_paths)
    all_ranked_nodes = remove_nodes_from_subsequent_ranks_preserve_empty(all_ranked_nodes)

    final_rows = create_final_ranking_using_sir_preserve_indices(
        all_ranked_nodes, G, lambda_value, output_file
    )

    print(f"  ✓ {dataset_name}: Total ranked rows written: {len(final_rows)}")

    # Step-6: fix rank gaps but preserve repeats
    reorder_ranks_preserve_repeats(output_file)


def main():
    candidates = [
        d for d in os.listdir(OUTPUT_ROOT)
        if os.path.isdir(os.path.join(OUTPUT_ROOT, d))
    ]

    if not candidates:
        print("No dataset subfolders found in outputs.")
        return

    for dataset_name in sorted(candidates):
        process_one_dataset(dataset_name)


if __name__ == "__main__":
    main()