# -*- coding: utf-8 -*-
"""
Final cleaned version
@author: Javaria
"""

import os
import glob
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

# ==========================
# CONFIGURATION
# ==========================

BASE_OUTPUTS_DIR = r"......\Attribute_codes\outputs_final_last"
GR_ROOT_DIR = os.path.dirname(BASE_OUTPUTS_DIR)

GAMMA = 1.0
MAX_TIME_STEPS = 100
NUM_SIMULATIONS = 100

# ==========================
# UTILITY FUNCTIONS
# ==========================

def find_single_file(patterns, folder):
    for pat in patterns:
        files = glob.glob(os.path.join(folder, pat))
        if len(files) >= 1:
            return files[0]
    return None


def find_gr_for_network(network_name, root_folder):
    gr_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for f in filenames:
            if f.lower().endswith(".gr"):
                gr_files.append(os.path.join(dirpath, f))

    for f in gr_files:
        if os.path.basename(f) == f"{network_name}.gr":
            return f

    for f in gr_files:
        if network_name in os.path.basename(f):
            return f

    return gr_files[0] if len(gr_files) == 1 else None


def read_gr_file(filename):
    G = nx.Graph()
    node_order = []
    seen = set()

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('a'):
                _, u, v = line.split()
                u, v = int(u), int(v)
                G.add_edge(u, v)
                for n in (u, v):
                    if n not in seen:
                        seen.add(n)
                        node_order.append(n)

    return G, node_order


def calculate_beta_c_from_graph(G):
    deg = np.array([d for _, d in G.degree()], dtype=float)
    if len(deg) == 0:
        return 0.0
    return deg.mean() / max(np.mean(deg ** 2) - deg.mean(), 1e-12)


# ==========================
# SIR MODEL
# ==========================

def simulate_sir_model(
    G,
    lambda_value,
    gamma=1.0,
    max_time_steps=100,
    num_simulations=100,
    seed_base=4
):
    """
    ORIGINAL spreading code – LOGIC UNCHANGED
    """

    starting_nodes = sorted(G.nodes())

    average_spreading_abilities = []

    for i, initial_node in enumerate(starting_nodes):

        np.random.seed(seed_base + i)

        cumulative_spreading_at_each_time_step = []

        for _ in range(num_simulations):
            infected_nodes = {initial_node}
            recovered_nodes = set()
            cumulative_spreading_simulations = []

            for _ in range(max_time_steps):
                newly_infected = set()
                newly_recovered = set()

                for node in infected_nodes:
                    for neighbor in sorted(G.neighbors(node)):
                        if (neighbor not in infected_nodes and
                            neighbor not in recovered_nodes and
                            np.random.random() < lambda_value):
                            newly_infected.add(neighbor)

                for node in list(infected_nodes):
                    if np.random.random() < gamma:
                        newly_recovered.add(node)
                        infected_nodes.remove(node)

                infected_nodes.update(newly_infected)
                recovered_nodes.update(newly_recovered)

                cumulative_spreading_simulations.append(
                    len(infected_nodes) + len(recovered_nodes)
                )

            cumulative_spreading_at_each_time_step.append(
                cumulative_spreading_simulations
            )

        avg_spread = np.mean(cumulative_spreading_at_each_time_step, axis=0)
        average_spreading_abilities.append(avg_spread)

    # 🔑 SAFE adapter for BIG code (NO math change)
    return {
        node: avg
        for node, avg in zip(starting_nodes, average_spreading_abilities)
    }



def save_spreading_for_lambdas(net_folder, G, node_order, lambda_values):

    for lam in lambda_values:
        print(f"  [SIR] λ = {lam:.4f}")
        sir = simulate_sir_model(G, lam, GAMMA, MAX_TIME_STEPS, NUM_SIMULATIONS)

        df = pd.DataFrame({
            "Starting Node": node_order,
            "Average Spreading Ability": [sir[n][-1] for n in node_order]
        })

        out_path = os.path.join(
            net_folder,
            f"average_spreading_abilities_lambda_{lam:.4f}.csv"
        )
        df.to_csv(out_path, index=False)
        print("    -> Saved:", out_path)
def compute_jaccard(network_folder,
                    final_ranking_file,
                    avg_spread_ranking_file,
                    k_values=None):

    if k_values is None:
        k_values = range(5, 81, 5)

    avg_df = pd.read_csv(avg_spread_ranking_file)
    fin_df = pd.read_csv(final_ranking_file)

    # ---- identify column names safely ----
    avg_rank_col = 'Rank'
    avg_node_col = 'Starting Node'
    if avg_node_col not in avg_df.columns and 'Node' in avg_df.columns:
        avg_node_col = 'Node'

    fin_rank_col = 'Rank'
    fin_node_col = 'Node'
    if fin_node_col not in fin_df.columns and 'Starting Node' in fin_df.columns:
        fin_node_col = 'Starting Node'

    jaccard_values = []

    for k in k_values:
        avg_set = set(avg_df[avg_df[avg_rank_col] <= k][avg_node_col])
        fin_set = set(fin_df[fin_df[fin_rank_col] <= k][fin_node_col])

        intersection = len(avg_set & fin_set)
        union = len(avg_set | fin_set)

        jacc = intersection / union if union > 0 else 0.0

        jaccard_values.append((k, jacc))
        print(f"  [Jaccard] Top {k}: {jacc:.4f}")

    jacc_df = pd.DataFrame(
        jaccard_values,
        columns=['Top K', 'Jaccard Similarity']
    )

    out_path = os.path.join(network_folder, "jaccard_similarities.csv")
    jacc_df.to_csv(out_path, index=False)
    print(f"    -> Saved Jaccard similarities to {out_path}")

    return jacc_df

# ==========================
# RANK COMPARISON
# ==========================

def compare_rankings_kendall(avg_file, final_file, lam, beta_c, beta_min, beta_max):

    avg = pd.read_csv(avg_file)
    fin = pd.read_csv(final_file)

    avg["Dense Rank"] = avg["Average Spreading Ability"].rank(
        method="dense", ascending=False
    ).astype(int)

    merged = avg.merge(
        fin[["Node", "Rank"]],
        left_on="Starting Node",
        right_on="Node",
        how="inner"
    )

    tau, p = kendalltau(merged["Dense Rank"], merged["Rank"])

    return {
        "Lambda": lam,
        "Kendall Tau": tau,
        "p-value": p,
        "beta_c": beta_c,
        "beta_min": beta_min,
        "beta_max": beta_max
    }


def compute_kendall_for_all_lambdas(net_folder, final_file, lambda_values,
                                    beta_c, beta_min, beta_max):

    rows = []

    for lam in lambda_values:
        avg_file = os.path.join(
            net_folder,
            f"average_spreading_abilities_lambda_{lam:.4f}.csv"
        )
        if not os.path.exists(avg_file):
            continue

        rows.append(
            compare_rankings_kendall(
                avg_file, final_file, lam, beta_c, beta_min, beta_max
            )
        )

    if rows:
        out = os.path.join(net_folder, "kendall_tau_summary_results.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print("    -> Saved:", out)


# ==========================
# MAIN DRIVER
# ==========================

def main():

    for network_name in os.listdir(BASE_OUTPUTS_DIR):

        net_folder = os.path.join(BASE_OUTPUTS_DIR, network_name)
        if not os.path.isdir(net_folder):
            continue

        print("\n==============================")
        print("Processing:", network_name)

        final_ranking = find_single_file(
            ["final_ranking*.csv", "final ranking*.csv"],
            net_folder
        )
        if not final_ranking:
            print("  ❌ No final ranking file")
            continue

        gr_file = find_gr_for_network(network_name, GR_ROOT_DIR)
        if not gr_file:
            print("  ❌ No .gr file")
            continue

        G, node_order = read_gr_file(gr_file)
        beta_c = calculate_beta_c_from_graph(G)

        beta_min, beta_max = 0.5 * beta_c, 1.5 * beta_c
        lambda_values = np.linspace(beta_min, beta_max, 10)

        save_spreading_for_lambdas(net_folder, G, node_order, lambda_values)
        avg_spread_rank_file = find_single_file(
            ["average_spreading_ranking*.csv"],
            net_folder
        )

        if avg_spread_rank_file:
            compute_jaccard(
                net_folder,
                final_ranking,
                avg_spread_rank_file
            )
        else:
            print("  ⚠️ No average_spreading_ranking file found, skipping Jaccard.")

        compute_kendall_for_all_lambdas(
            net_folder,
            final_ranking,
            lambda_values,
            beta_c,
            beta_min,
            beta_max
        )

        pd.DataFrame([{
            "network": network_name,
            "beta_c": beta_c,
            "beta_min": beta_min,
            "beta_max": beta_max
        }]).to_csv(
            os.path.join(net_folder, "beta_info.csv"),
            index=False
        )

    print("\n=== ALL DONE SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
