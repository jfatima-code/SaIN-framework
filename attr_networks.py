

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 15:22:51 2024

@author: DELL
"""

#constraint coefficient and second order centrality are manually multiplied with -1, as for these 2 attributes smaller the value greater the improtance

import networkx as nx
import pandas as pd
import math
import numpy as np
import time
# Function to read graph from a GR file
def read_gr_file(filename):
    G = nx.Graph()
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('a'):
                _, node1, node2 = line.split()
                G.add_edge(node1, node2)
    return G

# Function to calculate degree centrality
def calculate_degree_centrality(graph):
    degree_centrality = nx.degree_centrality(graph)
    return degree_centrality
print("test")
#__________________________________________________
def calculate_clustered_local_degree(graph):
    cld_scores = {}
    for node in graph.nodes:
        cci = calculate_clustering_coefficient(graph, node)
        cld_score = (1 + cci) * sum(graph.degree(j) for j in graph.neighbors(node))
        cld_scores[node] = cld_score
    return cld_scores
#________________________________________________________________________________
def leverage_centrality(graph, node):
    neighbors = set(graph.neighbors(node))
    degree_i = graph.degree(node)
    leverage_centrality = 0
    
    for neighbor in neighbors:
        degree_j = graph.degree(neighbor)
        leverage_centrality += (degree_i - degree_j) / (degree_i + degree_j)
    
    leverage_centrality *= 1 / degree_i
    
    return leverage_centrality



#_____________________________________________________________________
#4 Isolating centrality
def isolating_centrality(graph, node):
    neighbors = set(graph.neighbors(node))
    min_degree = min(dict(graph.degree()).values())
    min_degree_nodes = {n for n, d in graph.degree() if d == min_degree}
    isolating_centrality = len(neighbors.intersection(min_degree_nodes)) * graph.degree(node)
    return isolating_centrality

#_______________________________________________________________________
def calculate_node_distances(G, radii=None):
    """
    Calculate node distances with or without a given radius.
    """
    dij_values = {}
    for r in radii:
        dij_values[r]={}
        for node in G.nodes:
            shortest_path_lengths = nx.single_source_shortest_path_length(G, node, cutoff=r)
            dij_values[r][node] = shortest_path_lengths
    
    return dij_values
#____________________________________________________
#5 global self measure GSM
def global_self_measure(graph, node, distances):
    num_nodes = graph.number_of_nodes()
    
    # Calculate k-shell index using core number
    core_numbers = nx.core_number(graph)
    ks_i = core_numbers[node]
    full_graph_distances= distances[None]
    #k_shell_values = core_numbers.values()
    
    # Calculate Self Influence (SI)
    si = math.exp(ks_i / num_nodes)
    
    # Calculate Global Influence (GI)
    gi = 0
    for j in graph.nodes():
        if j != node:
            try:
                dij = full_graph_distances[node][j]
                gi += core_numbers[j] / dij
            except nx.NetworkXNoPath:
                pass
    
    # Calculate Global Self-Measure (GSM)
    gsm = si * gi
    
    
    return gsm

#_________________________________________________________________________________

#________________________________________________________________________________________
def calculate_H_GSM_values_1(G, distances):
    # Calculate K shell values
    Ks_values = nx.core_number(G)

    # Calculate degree centrality for each node
    DC_values = dict(nx.degree(G))
    full_graph_distances = distances[None]
    # Calculate improved self influence (ISI) values
    n = len(G.nodes)
    ISI_values_1 = {}
    for node in G.nodes:
        ISI_values_1[node] = np.exp((Ks_values[node] * DC_values[node]) / n)

    # Calculate average ISI across all nodes
    avg_ISI_1 = np.mean(list(ISI_values_1.values()))

    # Calculate distances between nodes
    #dij_values = {}
    #for node in G.nodes:
   # dij_values = full_graph_distances[None]

    # Calculate IGI values for each node
    IGI_values_1 = {}
    for node in G.nodes:
        IGI_sum_1 = 0
        for j in G.nodes():
            if j != node:
                dij_values=full_graph_distances[j][node]
                IGI_sum_1 += ISI_values_1[j] / (dij_values** (math.ceil(math.log2(avg_ISI_1))))
        IGI_values_1[node] = IGI_sum_1

    # Calculate H-GSM values for each node
    H_GSM_values_1 = {}
    for node in G.nodes:
        H_GSM_values_1[node] = ISI_values_1[node] * IGI_values_1[node]

    return H_GSM_values_1
print("H_GSM_values_1")
#_______________________________________________________________________________
#8 Local and global centrality

def local_global_centrality(graph, node, distances, alpha=0.5):
    num_nodes = graph.number_of_nodes()
    degree_i = graph.degree(node)
    full_graph_distances = distances[None]
    # Calculate Local Influence (LI)
    li = degree_i / num_nodes
    
    # Calculate Global Influence (GI)
    gi = 0
    for j in graph.nodes():
        if j != node:
            try:
                dij = full_graph_distances[node][j]
                gi += (math.sqrt(graph.degree(j) + alpha)) / dij
            except nx.NetworkXNoPath:
                continue
    
    # Calculate Local-Global Centrality (LGC)
    lgc = li * gi
    
    return lgc
print("lgc")
#______________________________________________________________________________________
def calculate_pagerank_centrality(G):
    
    pagerank_centrality = nx.pagerank(G)
    return pagerank_centrality
print(" pagerank_centrality")
#________________________________________________________________________________________
#10 coreness centrality
def coreness_centrality(graph, node, k_shell_values ):
    
    #print(core_numbers)
    neighborhood = list(graph.neighbors(node))
    coreness_cent = sum(k_shell_values[j] for j in neighborhood)
    return coreness_cent
print("coreness_cent")
#_______________________________________________________________________________________
#11 extende coreness uses formula of coreness above
def extended_coreness(graph, k_shell_values):
    extended_coreness_values = {}
    for node in graph.nodes():
        neighborhood = list(graph.neighbors(node))
        extended_coreness = sum(coreness_centrality(graph, j, k_shell_values) for j in neighborhood)
        extended_coreness_values[node] = extended_coreness
    return extended_coreness_values
print("extended_coreness_values")

#______________________________________________________________________________________
#12 global relative change dgree
def global_relative_change_avg_degree(graph, node):
    avg_degree = sum(dict(graph.degree()).values()) / len(graph)
    
    graph_v_removed = graph.copy()
    graph_v_removed.remove_node(node)
    
    avg_degree_v_removed = sum(dict(graph_v_removed.degree()).values()) / len(graph_v_removed)
    
    global_relative_change = abs((avg_degree_v_removed - avg_degree)) / avg_degree
    return global_relative_change
print("global_relative_change")

#__________________________________________________________________________________
def calculate_clustering_coefficient(graph, node):
   
    return nx.clustering(graph, node)
print("clustering_coefficient")
#______________________________________________________

#14 Global relative change in avg clustering

def average_clustering_coefficient(graph):
    # Directly use NetworkX's efficient implementation
    return nx.average_clustering(graph)

def global_relative_change_clustering(graph, node):
    # Calculate the average clustering coefficient for the original graph
    acc_g = average_clustering_coefficient(graph)
    
    # Create a subgraph without the specific node instead of copying and removing
    subgraph_v_removed = graph.subgraph(graph.nodes - [node])
    
    # Calculate the average clustering coefficient for the subgraph
    acc_g_v = average_clustering_coefficient(subgraph_v_removed)
    
    # Compute the global relative change
    global_relative_change = abs((acc_g_v - acc_g)) / acc_g
    
    return global_relative_change

print("global_relative_change")
#____________________________________________________________________________________


#______________________________________________________________________________________
#19 calculate DIL centrality

def calculate_dil_centrality(graph):
    dil_centrality = {}

    for node in graph.nodes:
        dil_centrality[node] = graph.degree(node)  # Initialize with node's degree

        for neighbor in graph.neighbors(node):
            ki = graph.degree(node)
            kj = graph.degree(neighbor)

            # Calculate p for the current edge (node, neighbor)
            p = 0
            for second_neighbor in graph.neighbors(neighbor):
                if graph.has_edge(node, second_neighbor):
                    p += 1

            # Calculate lambda and U
            lambda_value = (p / 2) + 1
            U = (ki - p - 1) * (kj - p - 1)

            # Calculate w for the current edge (node, neighbor)
            Ieij = U / lambda_value
            wij = Ieij * (ki - 1) / (ki + kj - 2)

            # Update the dil centrality for the current node
            dil_centrality[node] += wij
            
    return dil_centrality
#______________________________________________________________________________
def calculate_I_GSM_values(G,distances):
    DC_values = dict(nx.degree(G))
    full_graph_distances = distances[None]
    # Calculate improved self influence (ISI) values
    n = len(G.nodes)
    ISI_values = {}
    for node in G.nodes:
        ISI_values[node] = np.exp(( DC_values[node]) / n)

    
    #print(avg_ISI)
    # Calculate distances between nodes
   # dij_values = calculate_node_distances(G)
    # Calculate average ISI across all nodes
    avg_DC = np.mean(list(DC_values.values()))
    # Calculate IGI values for each node
    IGI_values = {}
    for node in G.nodes:
        IGI_sum = 0
        for j in G.nodes():
            if j != node:
                dij_values= full_graph_distances[j][node]
                IGI_sum += DC_values[j] / (dij_values ** (math.ceil(math.log2(avg_DC))))
        IGI_values[node] = IGI_sum
    
    
    # Calculate H-GSM values for each node
    I_GSM_values = {}
    for node in G.nodes:
        I_GSM_values[node] = ISI_values[node] * IGI_values[node]
    
    return I_GSM_values
#_________________________________________________________________________________________

#_________________________________________________________________________________________
def gravity_index_centrality(graph, node, distances, r=2):
    ks_i = nx.core_number(graph)[node]
    distances= calculate_node_distances(graph, radii=[2])
    node_distances=distances[r][node]
    # Get neighborhood set within radius r
    neighborhood_set = set(node_distances.keys())
    gravity_index = 0
    
    for j in graph.nodes():
        if j != node and j in neighborhood_set:
            try:
                dij = node_distances[j]
                ks_j = nx.core_number(graph)[j]
                gravity_index += (ks_i * ks_j) / (dij ** 2)
            except KeyError:
                pass
    
    return gravity_index

#______________________________________________________________________________________

#____________________________________________________________________________________

#__________________________________________________________________________________________________
#24 Gravity model
# Calculate gravity model centrality for each node
def gravity_model_centrality(graph, distances):
    gravity_centrality_values = {}
    
    # Get connected components of the graph
    components = list(nx.connected_components(graph))
    full_graph_distances = distances[None]
    for component in components:
        for node in component:
            gravity_centrality = 0
            ki = len(list(graph.neighbors(node)))  # Degree of node i
            
            for j in component:
                if j != node:
                    try:
                        dij = full_graph_distances[node][j]  # Shortest distance between i and j
                        kj = len(list(graph.neighbors(j)))  # Degree of node j
                        
                        gravity_centrality += (ki * kj) / dij**2
                    except nx.NetworkXNoPath:
                        continue  # Skip this pair of nodes if no path exists
            
            gravity_centrality_values[node] = gravity_centrality
    
    return gravity_centrality_values

#_____________________________________________________________________________________



def calculate_average_shortest_path_length(graph):
 
    avg_shortest_path_length = nx.average_shortest_path_length(graph)
    return avg_shortest_path_length


def local_gravity_model_centrality(graph, distances, truncation_radius=1):
    local_gravity_centrality_values = {}
    
    full_graph_distances = distances[None]
    
    for node in graph.nodes():
        local_gravity_centrality = 0
        ki = len(list(graph.neighbors(node)))  # Degree of node i
        
        for j in graph.nodes():
            if j != node:
                try:
                    dij = full_graph_distances[node][j]  # Use precomputed distance between i and j
                    
                    if dij <= truncation_radius:  # Truncation radius R
                        kj = len(list(graph.neighbors(j)))  # Degree of node j
                        local_gravity_centrality += (ki * kj) / (dij ** 2)
                except KeyError:
                    pass
        
        local_gravity_centrality_values[node] = local_gravity_centrality
    
    return local_gravity_centrality_values
#_________________________________________________________________________________
def calculate_laplacian_centrality(graph):
    laplacian_centrality = {}
    
    for node in graph.nodes:
        neighbors = set(graph.neighbors(node))
        degree_i = graph.degree(node)
        sum_values = 0
        
        for j in neighbors:
            degree_j = graph.degree(j)
            sum_values += degree_j
        
        centrality_score_1 = (degree_i ** 2) + degree_i + (2 * sum_values)
        laplacian_centrality[node] = centrality_score_1
    
    return laplacian_centrality
#__________________________________________________________________________________
def closeness_centrality_formula(graph, distances):
    closeness_centralities = {}
    full_graph_distances = distances[None]
    for node in graph.nodes:
        
        shortest_paths = full_graph_distances[node]
        total_distance = sum(shortest_paths.values())
        closeness_centrality = 1 / total_distance
        closeness_centralities[node] = closeness_centrality
    return closeness_centralities
#___________________________________________________________________________________

#___________________________________________________________________________________________
# 29 semi local / local centrality
# Calculate the set S2(u) for each node u
def calculate_S2_sets_2(graph):
    S2_sets_2 = {}
    
    for node in graph.nodes():
        S2_set_2 = set()  # Initialize an empty set
        
        # Calculate S2(u) by considering neighbors and their neighbors
        for neighbor in graph.neighbors(node):
            if neighbor != node:  # Exclude the node itself
                S2_set_2.add(neighbor)  # Add direct neighbors
            for neighbor2 in graph.neighbors(neighbor):
                if neighbor2 != node:  # Exclude the node itself
                    S2_set_2.add(neighbor2)  # Add neighbors of neighbors
        
        S2_sets_2[node] = S2_set_2
    
    return S2_sets_2

# Calculate N(u) for each node
def calculate_N_u(graph, S2_sets_2):
    N_u_values = {}
    
    for node in graph.nodes():
        N_u_values[node] = len(S2_sets_2[node])
    
    return N_u_values

# Calculate Q(w) for each node
def calculate_Q_w(graph, N_u_values):
    Q_w_values = {}
    
    for node in graph.nodes():
        Q_w_sum = 0
        
        # Calculate Q(w) for node w
        for neighbor_w in graph.neighbors(node):
            Q_w_sum += N_u_values[neighbor_w]
        
        Q_w_values[node] = Q_w_sum
    
    return Q_w_values

# Calculate LC(v) for each node
def calculate_semi_local_centrality(graph, Q_w_values):
    semi_local_centralities = {}
    
    for node in graph.nodes():
        lc_v = 0
        
        # Calculate LC(v) for node v
        for neighbor_v in graph.neighbors(node):
            lc_v += Q_w_values[neighbor_v]
        
        semi_local_centralities[node] = lc_v
    
    return semi_local_centralities

#______________________________________________________________________________
# Define your S2 function
def S2(G, node):
    # Initialize the set S2(u)
    s2_set = set()
    
    # Add neighbors of u to S2(u)
    neighbors = set(G.neighbors(node))
    s2_set.update(neighbors)
    
    # Add neighbors' neighbors (excluding u) to S2(u)
    for neighbor in neighbors:
        s2_set.update(G.neighbors(neighbor))
    
    # Remove u from S2(u)
    s2_set.discard(node)
    
    return s2_set

def calculate_S2_sets(G):
    S2_sets = {}
    for node in G.nodes():
        S2_sets[node] = S2(G, node)
    return S2_sets

def calculate_clustering_coefficients(G):
    return {node: calculate_clustering_coefficient(G, node) for node in G.nodes()}

def calculate_Q(G, alpha, node, clustering_coefficients):
    # Calculate S2(node) for the specific node
    S2_node = S2(G, node)
    
    # Initialize Q(node)
    Q_node = alpha * len(S2_node)

    # Calculate the sum of local clustering coefficients for nodes in S2(node)
    sum_cp = sum(clustering_coefficients.get(p, 0.0) for p in S2_node)

    # Add (1 - alpha) * sum_cp to Q(node)
    Q_node += (1 - alpha) * sum_cp

    return Q_node

def calculate_centrality(G, alpha):
    centrality_values = {}
    
    # Precompute S2 sets and clustering coefficients
    S2_sets = calculate_S2_sets(G)
    clustering_coefficients = calculate_clustering_coefficients(G)
    
    for node in G.nodes():
        # Initialize Centrality(i)
        centrality_i = 0.0

        # Get neighbors of node i
        neighbors_i = list(G.neighbors(node))

        # Calculate Centrality(i) by summing Q(i) over all neighbors
        for neighbor_j in neighbors_i:
            centrality_i += calculate_Q(G, alpha, neighbor_j, clustering_coefficients)
        
        centrality_values[node] = centrality_i

    return centrality_values
#____________________________________________________________________________
def calculate_CLC_values(G, calculate_clustering_coefficient, semi_local_centralities):
    clc_values = {}
    for node in G.nodes():
        clc_values[node] = math.exp(- calculate_clustering_coefficient(G, node)) * semi_local_centralities[node]
        # Uncomment the following line if you want to print CLC values for each node
        # print(f"Node {node}: CLC({node}) = {clc_values[node]}")
    return clc_values


#________________________________________________________________________________
def calculate_eigenvector_centrality(G):
    
    eigenvector_centrality  = nx.eigenvector_centrality(G, max_iter=1000)
    return eigenvector_centrality
#______________________________________________________________________________
def calculate_extended_degrees(G, delta=0.6):
 
    # Create a dictionary to store the extended degree for each node
    extended_degrees = {}

    # Calculate the extended degree for each node
    for node in G.nodes():
        k_u = G.degree(node)
        sum_k_v = sum(G.degree(neighbor) for neighbor in G.neighbors(node))
        extended_degree = delta * k_u + (1 - delta) * sum_k_v
        extended_degrees[node] = extended_degree
    
    return extended_degrees


#_______________________________________________________________________________
def calculate_p_values(G, delta=0.5):
      
    # Find the maximum extended degree (k_ex)max
   # max_extended_degree = max(calculate_extended_degrees)
    
    # Initialize the value of p for each node
    p_values = {}

    # Set the initial value of p for each node
    for node in G.nodes():
        p_values[node] = 1

    # Initialize Gp as a copy of the original graph G
    Gp = G.copy()

    # Calculate p for each node using the iterative process
    p = 1
    
    while Gp.number_of_nodes() > 0:
        # Calculate the extended degree for each node in Gp
        extended_degrees = calculate_extended_degrees(Gp, delta)
        for node in Gp.nodes():
            extended_degree = extended_degrees[node]
            Gp.nodes[node]['extended_degree'] = extended_degree

        # Find the set Sp with nodes having the minimum extended degree
        min_extended_degree = min(nx.get_node_attributes(Gp, 'extended_degree').values())
        Sp = [node for node in Gp.nodes() if Gp.nodes[node]['extended_degree'] == min_extended_degree]

        # Update p for nodes in Sp
        for node in Sp:
            p_values[node] = p

        # Remove nodes in Sp from Gp to create the next graph
        Gp.remove_nodes_from(Sp)

        # Increment the iteration number
        p += 1
    
    return p_values

#___________________________________________________________________________________

def calculate_hcc(G_original, calculate_extended_degrees, calculate_p_values):
    
    p_values = calculate_p_values(G_original, delta=0.5)

    extended_degrees = calculate_extended_degrees(G_original, delta=0.5)
    print(extended_degrees)
    # Find the maximum value of p (max p)
    max_p = max(p_values.values())

    # Find the maximum extended degree (k_ex)max
    max_extended_degree = max(extended_degrees.values())

    # Create a dictionary to store HCC
    hcc_dict = {}

    # Calculate HCC for each node
    for node in G_original.nodes():

        k_ex_u = extended_degrees[node]

        p_u = p_values.get(node, 0)

        hcc_value = (k_ex_u / max_extended_degree) + (p_u / max_p)

        hcc_dict[node] = hcc_value

    return hcc_dict
#_____________________________________________________________

def calculate_ehcc(G_original, calculate_extended_degrees, calculate_p_values):
    
    # Calculate HCC values
    hcc_dict = calculate_hcc(
        G_original,
        calculate_extended_degrees,
        calculate_p_values
    )
    
    # Initialize dictionary to store EHCC values
    ehcc_values = {}
    
    # Calculate EHCC for each node
    for node in G_original.nodes():

        ehcc_value = hcc_dict[node] + sum(
            hcc_dict[neighbor]
            for neighbor in G_original.neighbors(node)
        )

        ehcc_values[node] = ehcc_value
    
    return ehcc_values

#______________________________________________________________________________
def calculate_mdd(G, lambda_val=0.7):
    
    G_C = G.copy()
    
    # Initialize values for kr, ke, and lambda
    kr = {node: len(list(G_C.neighbors(node))) for node in G.nodes()}  # Residual degree
    ke = {node: 0 for node in G.nodes()}  # Exhausted degree

    # Store the removed nodes and their km values
    removed_nodes_km = {}

    # Main loop
    step = 0
    while G_C.number_of_nodes() > 0:
        # Calculate km values for each node
        km = {node: kr[node] + lambda_val * ke[node] for node in G_C.nodes()}

        # Find nodes with minimum km values
        min_km_value = min(km.values())
        nodes_to_remove = [node for node, km_value in km.items() if km_value == min_km_value]

        # Update ke values for neighbors of removed nodes
        for removed_node in nodes_to_remove:
            for neighbor in G.neighbors(removed_node):
                if neighbor in ke:
                    ke[neighbor] += 1
        # Update kr values for neighbors of removed nodes
        for removed_node in nodes_to_remove:
            for neighbor in G_C.neighbors(removed_node):
                if neighbor in kr:
                    kr[neighbor] -= 1

        # Store the km values for removed nodes at this step
        for removed_node in nodes_to_remove:
            removed_nodes_km[removed_node] = km[removed_node]

        # Remove nodes with minimum km values
        G_C.remove_nodes_from(nodes_to_remove)

        step += 1

    return removed_nodes_km
#______________________________________________________
def calculate_density_centrality(graph, distances, r=2):
    density_centrality = {}
    pi = 22 / 7  # Value of pi
    dist=distances[None]
    # Iterate over connected components
    for component in nx.connected_components(graph):
        # Calculate density centrality for each node in the connected component
        for node in component:
            shortest_paths = dist[node]
            eligible_nodes = [target for target, distance in shortest_paths.items() if distance is not None and distance <= r]

            sum_values = 0

            for j in eligible_nodes:
                if j != node:
                    degree_i = graph.degree(node)
                    shortest_path_length = shortest_paths[j]
                    sum_values += degree_i / (pi * (shortest_path_length ** 2))

            density_centrality[node] = sum_values
    
    return density_centrality

#_________________________________________________________________________________________________
def calculate_eccentricity_centrality(graph):
    eccentricity_centrality_dict = {}
    
    # Calculate eccentricity centrality for all nodes
    for node in graph.nodes():
        eccentricity = nx.eccentricity(graph, v=node)
        eccentricity_centrality = 1 / eccentricity
        eccentricity_centrality_dict[node] = eccentricity_centrality
    
    return eccentricity_centrality_dict
#_____________________________________________________________________________________
def calculate_pij(graph):
    sum_eij = {node: sum(1 for _ in graph.neighbors(node)) for node in graph.nodes}
    pij_matrix = {}
    for node in graph.nodes:
        pij_matrix[node] = {neighbor: 1 / sum_eij[node] for neighbor in graph.neighbors(node)}
    return pij_matrix


def calculate_constraint_coefficient(graph, pij_matrix):
    # Create a mapping from node labels to integer indices
    node_to_index = {node: i for i, node in enumerate(graph.nodes())}
    
    constraint_coefficients = {}
    for node in graph.nodes():
        neighborhood = set(graph.neighbors(node))
        coefficient_sum = 0
        for j in neighborhood:
            indirect_neighbors = set(graph.neighbors(j)) - {node}
            q_sum = 0
            for q in indirect_neighbors:
                if node in pij_matrix and q in pij_matrix[node]:
                    pq = pij_matrix[node][q]
                    piq = pij_matrix[q][node]
                    pqj = pij_matrix[node][j]
                    q_sum += (piq * pqj)
            if node in pij_matrix and j in pij_matrix[node]:
                coefficient_sum += (pij_matrix[node][j] + q_sum)**2
        constraint_coefficients[node] = coefficient_sum
    return constraint_coefficients



#______________________________________________________________________________________
def calculate_node_h_index(graph, node):
    degrees = sorted([graph.degree(neighbor) for neighbor in graph.neighbors(node)], reverse=True)
    h_index = max([min(d, i) for i, d in enumerate(degrees, 1)])
    return h_index

def calculate_node_hv_index(graph, node, node_h_indices):
    h_index = node_h_indices[node]
    hv_index = sum(graph.degree(neighbor) for neighbor in graph.neighbors(node) if graph.degree(neighbor) >= h_index)
    return hv_index

def calculate_node_hvgc_index(graph, node, node_hv_indices, constraint_coefficient, distances, R=1):
    hvgc_index = 0
    node_distances = distances[None]  # Use precomputed distances for radius R
    for other_node in graph.nodes():
        if other_node != node and node_distances[node][other_node] <= R:
            dij = node_distances[node][other_node]
            hv_product = node_hv_indices[node] * node_hv_indices[other_node]
            hvgc_index += math.exp(-constraint_coefficient[node]) * (hv_product / dij**2)
    return hvgc_index

#___________________________________________________________________________________


#_________________________________________________________________

#__________________________________________________________________________________
def calculate_PE(graph):
    """
    Calculate the PE (Prpagation Entropy) measure for each node in the graph.

    Parameters:
        graph (networkx.Graph): The input graph.

    Returns:
        dict: A dictionary containing the PE value for each node.
    """
    def calculate_CN(graph, node):
        """
        Calculate the CN (Common Neighbors) measure for a given node.

        Parameters:
            graph (networkx.Graph): The input graph.
            node: The node for which CN is calculated.

        Returns:
            float: The CN value for the node.
        """
        # Get the first-order neighbors
        neighbors = set(graph.neighbors(node))
        # Initialize a set to store second-order neighbors
        second_order_neighbors = set()
        
        # Iterate over first-order neighbors to find their neighbors (second-order neighbors)
        for neighbor in neighbors:
            # Add neighbors of neighbors to the second-order neighbors set
            second_order_neighbors.update(graph.neighbors(neighbor))
        
        # Exclude the node itself and its first-order neighbors from the set of second-order neighbors
        second_order_neighbors -= {node}
        second_order_neighbors -= neighbors
        
        # Calculate the number of first-order neighbors and second-order neighbors
        num_neighbors = len(neighbors)
        num_second_order_neighbors = len(second_order_neighbors)
        
        # Calculate the clustering coefficient of the node
        #clustering_coefficient = nx.clustering(graph, node)
        
        # Calculate the CN value using the formula
        cn = (num_second_order_neighbors + num_neighbors) / (1 + calculate_clustering_coefficient(graph, node))
           
            
        return cn

    # Calculate CN for each node
    nodes_CN = {node: calculate_CN(graph, node) for node in graph.nodes()}
    
    # Calculate total CN
    total_CN = sum(nodes_CN.values())
    
    # Calculate I(i) for each node
    nodes_I = {node: CN / total_CN for node, CN in nodes_CN.items()}
    
    # Calculate PE(i) for each node
    nodes_PE = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        PE = -sum([nodes_I[n] * math.log(nodes_I[n]) for n in neighbors if n in nodes_I])
        nodes_PE[node] = PE
    
    return nodes_PE
#_________________________________________________________________________________
# HC attribute 
def compute_H(G):
    def compute_k_shell_sum(G):
        shells = nx.core_number(G)
        k_shell_sums = {node: sum(shells[n] for n in G.neighbors(node)) for node in G.nodes()}
        return k_shell_sums

    k_shell_sums = compute_k_shell_sum(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    H_values = []
    for node in G.nodes():
        H_values.append(k_shell_sums[node] * eigenvector_centrality[node])
    
    return H_values
#_________________________________________________________________________________
def volume_centrality_for_all_nodes(graph, h=2):
    """
    Compute the volume centrality for all nodes in the graph within distance h.

    Parameters:
    - graph: A NetworkX graph object.
    - h: The maximum distance within which neighbors are considered.

    Returns:
    - A dictionary containing volume centrality values for all nodes in the graph.
    """

    def bfs(graph, source, h):
        """
        Breadth-first search to find neighbors within distance h from the source node.

        Parameters:
        - graph: A NetworkX graph object.
        - source: The source node.
        - h: The maximum distance within which neighbors are considered.

        Returns:
        - A set containing nodes within distance h from the source node.
        """
        visited = set([source])
        queue = [(source, 0)]
        while queue:
            node, distance = queue.pop(0)
            if distance >= h:
                continue
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return visited

    volume_centrality = {}
    for node in graph.nodes():
        neighbors_within_h = bfs(graph, node, h)
        # Sum the degrees of neighbors within distance h
        volume_centrality[node] = sum(graph.degree(neighbor) for neighbor in neighbors_within_h) - graph.degree(node)

    return volume_centrality
#______________________________________________________________


def calculate_GIN(G, node, distances, alpha=1, beta=1):
    """
    Calculate GIN(i) for a given node i in the graph.

    Args:
    - G: NetworkX graph object
    - node: Node in the graph for which to calculate GIN(i)
    - alpha: Alpha value for SIM calculation
    - beta: Beta value for GIM calculation

    Returns:
    - GIN(i): Calculated GIN(i) value for the given node
    """
    def calculate_SIM(G, node, alpha=1):
        """
        Calculate SIM(i) using the given formula.

        Args:
        - G: NetworkX graph object
        - node: Node in the graph for which to calculate SIM(i)
        - alpha: Alpha value

        Returns:
        - SIM(i): Calculated similarity value for the given node
        """
        
        n = len(G.nodes)
        degree_i = G.degree(node)
        return math.exp((1 / n) * degree_i * alpha)

    def calculate_GIM(G, node, distances, beta=1):
        """
        Calculate GIM(i) for a given node i in the graph.

        Args:
        - G: NetworkX graph object
        - node: Node in the graph for which to calculate GIM(i)
        - beta: Beta value

        Returns:
        - GIM(i): Calculated GIM(i) value for the given node
        """
        GIM_i = 0
        full_graph_distances= distances[None]
        for other_node in G.nodes():
            if other_node != node:
                degree_j = G.degree(other_node)
                dij_values = full_graph_distances[node][other_node]
                GIM_i += degree_j * beta / dij_values
        return GIM_i
    SIM_i = calculate_SIM(G, node, alpha)
    GIM_i = calculate_GIM(G, node, distances, beta)
    return  SIM_i * GIM_i
#____________________________________________________________________
#this lap gravity attribute
def calculate_LGC(graph, laplacian_centrality_scores, distances):
    laplacian_centrality_scores = calculate_laplacian_centrality(graph)
    
    
    dist_val=distances[None]
    LGC_scores = {}
  #  num_nodes = len(graph.nodes)
    #avg_distance = nx.average_shortest_path_length(graph)
  #  avg_distance = calculate_average_shortest_path_length(graph)
    for node_i in graph.nodes:
        LGC_i = 0
      #  degree_i = graph.degree(node_i)
        LC_i = laplacian_centrality_scores[node_i]
        
        for node_j in graph.nodes:
            if node_i != node_j:
             #   degree_j = graph.degree(node_j)
                dij = dist_val[node_i][node_j]
                if dij <= 2:
                    LGC_i += (LC_i * laplacian_centrality_scores[node_j]) / (dij ** 2)
        
        LGC_scores[node_i] = LGC_i
    
    return LGC_scores
#__________________________________________________________________________________
def calculate_DNC(graph, alpha=1 ):
    DNC_scores = {}
    for node in graph.nodes():
        k_i = graph.degree(node)  # Degree of node i
        clustering_sum = sum(calculate_clustering_coefficient(graph, neighbor) for neighbor in graph.neighbors(node))
        DNC_scores[node] = k_i + alpha * clustering_sum
    return DNC_scores
#_______________________________________________________________________________________
def calculate_EIGSM_values(G, I_GSM_values):
    """
    Calculate the EIGSM values for all nodes in the graph.

    Parameters:
        G (networkx.Graph): The input graph.
        I_GSM_values (dict): A dictionary containing I_GSM values for nodes.

    Returns:
        dict: A dictionary containing the EIGSM values for all nodes in the graph.
    """
    EIGSM_values = {}
    for node in G.nodes:
        EIGSM_sum = sum(score[1] for j in G.neighbors(node) for score in I_GSM_values.items() if score[0] == j)
        EIGSM_values[node] = EIGSM_sum
    return EIGSM_values
#_____________________________________________________________________________
def calculate_societal_capital(graph):
    societal_capital_scores = {}
    
    for node in graph.nodes():
        k_i = graph.degree(node)  # Degree of node i
        neighbor_degrees_sum = sum(graph.degree(neighbor) for neighbor in graph.neighbors(node))
        societal_capital_scores[node] = k_i + neighbor_degrees_sum
    
    return societal_capital_scores
#_______________________________________________________________________________

def improved_gravity_index_centrality(graph, node, distances, r=2):
    ks_i = nx.core_number(graph)[node]
    
    gravity_index = 0
    dij_values=distances[r]
    # Extract the neighborhood set from the calculated distances
    neighborhood_set = set(dij_values[node].keys())
    
    for j in graph.nodes():
        if j != node and j in neighborhood_set:
            try:
                dij = distances[r][node][j]
                k_j = nx.degree(graph)[j]
                gravity_index += (ks_i * k_j) / (dij ** 2)
            except nx.NetworkXNoPath:
                pass
    
    return gravity_index
#____________________________________________________________________
def calculate_extended_improved_gravity_index_values(graph, distances):
    extended_improved_gravity_index_values = {}
    
    # Precompute gravity index centralities for all nodes to avoid recalculation
    gravity_index_centralities = {}
    for node in graph.nodes():
        gravity_index_centralities[node] = improved_gravity_index_centrality(graph, node, distances, r=2)
    
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        extended_improved_gravity_index = sum(gravity_index_centralities.get(neighbor, 0) for neighbor in neighbors)
        extended_improved_gravity_index_values[node] = extended_improved_gravity_index
    
    return extended_improved_gravity_index_values
#________________________________________________________________________
#41 MGCM
import statistics
# Function to calculate MCGM for a node i
def calculate_mcgm(node_i, graph, distances, r=2):
    # Calculate degree of each node
    degree_dict = dict(graph.degree())

    # Determine k-shell of each node
    k_shell_dict = nx.core_number(graph)

    # Compute eigenvector centrality of each node
    eigenvector_dict = nx.eigenvector_centrality(graph, max_iter=1000)
    dij_values = distances[None]
    # Find maximum values
    max_degree = max(degree_dict.values())
    max_k_shell = max(k_shell_dict.values())
    max_eigenvector = max(eigenvector_dict.values())

    # Calculate medians
    median_degree = statistics.median(degree_dict.values())
    median_k_shell = statistics.median(k_shell_dict.values())
    median_eigenvector = statistics.median(eigenvector_dict.values())

    # Calculate alpha
    alpha_numerator = max(median_degree / max_degree, median_eigenvector / max_eigenvector)
    alpha_denominator = median_k_shell / max_k_shell
    alpha = alpha_numerator / alpha_denominator

    # R: Maximum distance to consider
    R = 2  # You can set this according to your requirements

    mcgm_i = 0
    for node_j in graph.nodes():
        if node_i != node_j and dij_values[node_i][node_j] <= R :
            k_i = degree_dict[node_i]
            ks_i = k_shell_dict[node_i]
            e_i = eigenvector_dict[node_i]
            k_j = degree_dict[node_j]
            ks_j = k_shell_dict[node_j]
            e_j = eigenvector_dict[node_j]
            dij = dij_values[node_i][node_j]
            mcgm_i += ((k_i / max_degree) + (alpha * ks_i / max_k_shell) + (e_i / max_eigenvector)) * ((k_j / max_degree) + (alpha * ks_j / max_k_shell) + (e_j / max_eigenvector)) / dij**2
    return mcgm_i
#________________________________________________________




# Function to calculate GGC for node i
def calculate_GGC(G, node_i, distances, R=2):
    
    def calculate_S(G, node_i):
        clustering_coefficient_i = calculate_clustering_coefficient(G, node_i)
        k_i = G.degree(node_i)
        alpha = 2
        
        S_i = math.exp(-alpha * clustering_coefficient_i) * k_i
        return S_i
    ggc_i = 0
    dij_values = distances[None]
    for node_j in G.nodes():
        if node_i != node_j and dij_values[node_i][node_j] <= R:
            S_i = calculate_S(G, node_i)
            S_j = calculate_S(G, node_j)
            dij = dij_values[node_i][node_j]
            ggc_i += (S_i * S_j) / (dij ** 2)
    return ggc_i


#________________________________________________________________________


#___________________________________________________________
def calculate_k_shell_values(graph):
    return nx.core_number(graph)



   

# Function to calculate normalization factor
def calculate_betweenness(graph):
    return nx.betweenness_centrality(graph)



#__________________________________________________________________________
# Main function
start_time_total= time.process_time()
def main():
    # Load graph from GR file
    #filename = "aves-weaver-social-3.gr"
    
    filename = "ENZYMES_g10.gr"
    G = read_gr_file(filename)
    # Load network data from a CSV file (assuming the file has no header)
   # df_none = pd.read_csv(r'E:\\New folder\Attributes\Global and local centrality\Global and local centrality .csv', header=None)

    # Create a graph from the adjacency matrix
   # G = nx.from_pandas_adjacency(df_none)
    nx.draw(G, with_labels=True)
    radii= [2, None]
    distances= calculate_node_distances(G, radii)
    start_time_degree= time.process_time()
    
    # Calculate attributes
    degree_centrality = calculate_degree_centrality(G)
    end_time_degree= time.process_time()
    elapsed_time_degree= end_time_degree- start_time_degree
    print("elapsed_time_degree", elapsed_time_degree)
    print("****")
    #_____________________________________-
 
    start_time_leverage= time.process_time()
    leverage_centrality_values = {node: leverage_centrality(G, node) for node in G.nodes()}
    end_time_leverage= time.process_time()
    elapsed_time_leverage= end_time_leverage- start_time_leverage
    print("elapsed_time_leverage", elapsed_time_leverage)
    print("$$$$$")
    #___________________________________________________________
    start_time_isolating= time.process_time()
    isolating_centrality_values = {node: isolating_centrality(G, node) for node in G.nodes()} 
    print("iso")
    end_time_isolating= time.process_time()
    elapsed_time_isolating= end_time_isolating- start_time_isolating
    print("elapsed_time_isolating", elapsed_time_isolating)
    #_________________________________________________________________
    start_time_gsm= time.process_time()
    gsm_values = {node: global_self_measure(G, node,distances) for node in G.nodes()}  # Calculate GSM
    end_time_gsm= time.process_time()
    elapsed_time_gsm= end_time_gsm- start_time_gsm
    print("elapsed_time_gsm", elapsed_time_gsm)
    print("gsm")
    #___________________________________________________________________________________--

    #______________________________________________________________________________________
    start_time_H_GSM_values_1= time.process_time()
    H_GSM_values_1 = calculate_H_GSM_values_1(G, distances)
    end_time_H_GSM_values_1= time.process_time()
    elapsed_time_H_GSM_values_1= end_time_H_GSM_values_1- start_time_H_GSM_values_1
    print("elapsed_time_H_GSM_values_1", elapsed_time_H_GSM_values_1)
    print("hgsm")
    #________________________________________________________________________________________
    start_time_lgc= time.process_time()
    lgc_values = {node: local_global_centrality(G, node, distances) for node in G.nodes()}  # Calculate LGC values
    end_time_lgc= time.process_time()
    elapsed_time_lgc= end_time_lgc- start_time_lgc
    print("elapsed_time_lgc", elapsed_time_lgc)
    print("#####################")
    #________________________________________________________________________________________-
    start_time_lgc= time.process_time()
    pagerank_centrality_values = calculate_pagerank_centrality(G)  # Calculate PageRank centrality
    end_time_lgc= time.process_time()
    elapsed_time_lgc= end_time_lgc- start_time_lgc
    print("elapsed_time_lgc", elapsed_time_lgc)
    print("@@@@@@@@@@@@@@@@")
    #_____________________________________________________________________________________
   
    start_time_kshell= time.process_time()
    k_shell_values = calculate_k_shell_values(G)
    end_time_kshell= time.process_time()
    elapsed_time_kshell= end_time_kshell- start_time_kshell
    print("elapsed_time_kshell", elapsed_time_kshell)
   
    #_______________________________________________________
    start_time_coreness= time.process_time()
    coreness_centrality_values = {node: coreness_centrality(G, node, k_shell_values) for node in G.nodes()}
    end_time_coreness= time.process_time()
    elapsed_time_coreness= end_time_coreness- start_time_coreness
    print("elapsed_time_coreness", elapsed_time_coreness)
    print("core")
    #_________________________________________________________________________________________
    start_time_Ecoreness= time.process_time()
    extended_coreness_values = extended_coreness(G, k_shell_values)
    end_time_Ecoreness= time.process_time()
    elapsed_time_Ecoreness= end_time_Ecoreness- start_time_Ecoreness
    print("elapsed_time_Ecoreness", elapsed_time_Ecoreness)
    print("^^^^^^^^^^^^")
    #_______________________________________________________________________
    start_time_grad= time.process_time()
    global_relative_change_avg_degree_values = {node: global_relative_change_avg_degree(G, node) for node in G.nodes()}
    end_time_grad= time.process_time()
    elapsed_time_grad= end_time_grad- start_time_grad
    print("elapsed_time_grad", elapsed_time_grad)
    print("grad")
    #_________________________________________________________________________-
   # start_time_grac= time.process_time()
    global_relative_change_clustering_values = {node: global_relative_change_clustering(G, node) for node in G.nodes()}
    #end_time_grac= time.process_time()
    #elapsed_time_grac= end_time_grac- start_time_grac
    #print("elapsed_time_grac", elapsed_time_grac)
    #print("grac")
    #____________________________________________________________________________-
  
    start_time_clust= time.process_time() 
    clustering_coefficients_values = {node: calculate_clustering_coefficient(G, node)for node in G.nodes()}  # Calculate clustering coefficients
    end_time_clust= time.process_time()
    elapsed_time_clust= end_time_clust- start_time_clust
    print("elapsed_time_clust", elapsed_time_clust)
    print("clust")
    #___________________________________________________________________________________

    #____________________________________________________________________________________
    start_time_dil= time.process_time()
    dil_centrality_values = calculate_dil_centrality(G)
    end_time_dil= time.process_time()
    elapsed_time_dil= end_time_dil- start_time_dil
    print("elapsed_time_dil", elapsed_time_dil)
    print("dil")
    #____________________________________________________________________________________
    start_time_I_GSM= time.process_time()
    I_GSM_values = calculate_I_GSM_values(G, distances)
    end_time_I_GSM= time.process_time()
    elapsed_time_I_GSM= end_time_I_GSM- start_time_I_GSM
    print("elapsed_time_I_GSM", elapsed_time_I_GSM)
    print("IGSM")
    #________________________________________________________________________
    clustered_local_degree_values = calculate_clustered_local_degree(G)
   
    #_______________________________________________________________________
    start_time_gravity_index= time.process_time()
    gravity_index_values = {node: gravity_index_centrality(G, node,distances, r=2) for node in G.nodes()}
    end_time_gravity_index= time.process_time()
    elapsed_time_gravity_index= end_time_gravity_index- start_time_gravity_index
    print("elapsed_time_gravity_index", elapsed_time_gravity_index)
    print("gravity_index_values")
    #_______________________________________________________________________________
 
    #__________________________________________________________
    start_time_gravity_centrality= time.process_time()
    gravity_centrality_values = gravity_model_centrality(G, distances)
    end_time_gravity_centrality= time.process_time()
    elapsed_time_gravity_centrality= end_time_gravity_centrality- start_time_gravity_centrality
    print("elapsed_time_gravity_centrality", elapsed_time_gravity_centrality)
    print("grav")
    #______________________
    start_time_local_gravity= time.process_time()
    local_gravity_centrality_values = local_gravity_model_centrality(G, distances, truncation_radius=1)
    end_time_local_gravity= time.process_time()
    elapsed_time_local_gravity= end_time_local_gravity- start_time_local_gravity
    print("elapsed_time_local_gravity", elapsed_time_local_gravity)
    print("loc_g")
    
    #_____________________________________________________________
    start_time_local_gravity= time.process_time()
    laplacian_centrality_values = calculate_laplacian_centrality(G)
    end_time_local_gravity= time.process_time()
    elapsed_time_local_gravity= end_time_local_gravity- start_time_local_gravity
    print("elapsed_time_local_gravity", elapsed_time_local_gravity)
    print("lapla")
    #_______________________________________________________
    start_time_closeness= time.process_time()
    closeness_centralities = closeness_centrality_formula(G, distances)
    end_time_closeness= time.process_time()
    elapsed_time_closeness= end_time_closeness- start_time_closeness
    print("elapsed_time_closeness", elapsed_time_closeness)
    print("close")
    #______________________________________________________

    #____________________________________________________________---
    # Calculate S2(u) for each node
    start_time_semi= time.process_time()
    S2_sets = calculate_S2_sets(G)

    # Calculate N(u) for each node
    N_u_values = calculate_N_u(G, S2_sets)

    # Calculate Q(w) for each node
    Q_w_values = calculate_Q_w(G, N_u_values)

    # Calculate LC(v) for the graph
    semi_local_centralities = calculate_semi_local_centrality(G, Q_w_values)
    end_time_semi= time.process_time()
    elapsed_time_semi= end_time_semi- start_time_semi
    print("elapsed_time_semi", elapsed_time_semi)
    print("semi")
 #_______________________________________________________________________  
    # Example usage:
    start_time_centrality= time.process_time()
    alpha = 0.7  # Set your desired value for alpha

    # Calculate and print Centrality(i) for each node
    centrality_values = calculate_centrality(G, alpha)
    end_time_centrality= time.process_time()
    elapsed_time_centrality= end_time_centrality- start_time_centrality
    print("elapsed_time_centrality", elapsed_time_centrality)
    print("cent")
#_____________________________________________________________________________    
    start_time_mdd= time.process_time()
    mdd_values = calculate_mdd(G)
    end_time_mdd= time.process_time()
    elapsed_time_mdd= end_time_mdd- start_time_mdd
    print("elapsed_time_mdd", elapsed_time_mdd)
    print("mdd")
    #_____________________________________________________________-
    start_time_density= time.process_time()
    density_centralities = calculate_density_centrality(G, distances, r=2)
    end_time_density= time.process_time()
    elapsed_time_density= end_time_density- start_time_density
    print("elapsed_time_density", elapsed_time_density)
    print("dense")
    #_____________________________________________________________________
    start_time_eccentricity= time.process_time()
    eccentricity_centralities = calculate_eccentricity_centrality(G)
    end_time_eccentricity= time.process_time()
    elapsed_time_eccentricity= end_time_eccentricity- start_time_eccentricity
    print("elapsed_time_eccentricity", elapsed_time_eccentricity)
    print("ecc")
#_________________________________________________________
# Calculate extended degrees and p values
   # Calculate p values
    start_time_p_values= time.process_time()
    p_values = calculate_p_values(G, delta=0.5)
    end_time_p_values= time.process_time()
    elapsed_time_p_values= end_time_p_values- start_time_p_values
    print("elapsed_time_p_values", elapsed_time_p_values)
    print("p")

    # Calculate extended degrees
    start_time_ext_degree= time.process_time()
    extended_degrees = calculate_extended_degrees(G, delta=0.6)
    end_time_ext_degree= time.process_time()
    elapsed_time_ext_degree= end_time_ext_degree- start_time_ext_degree
    print("elapsed_time_ext_degree", elapsed_time_ext_degree)
    print("ext deg")
    #_______________________________________________
    # Calculate HCC values
    start_time_hcc= time.process_time()
    hcc_values = calculate_hcc(G, calculate_extended_degrees, calculate_p_values)
    end_time_hcc= time.process_time()
    elapsed_time_hcc= end_time_hcc- start_time_hcc
    print("elapsed_time_hcc", elapsed_time_hcc)
    print("hcc")
    #________________________________________
    # Calculate EHCC values
    start_time_ehcc= time.process_time()
    ehcc_values = calculate_ehcc(G, calculate_extended_degrees, calculate_p_values)
    end_time_ehcc= time.process_time()
    elapsed_time_ehcc= end_time_ehcc- start_time_ehcc
    print("elapsed_time_ehcc", elapsed_time_ehcc)
    print("ehcc")
#________________________________________________________________________
    start_time_eigenvector = time.process_time()
    eigenvector_centrality = calculate_eigenvector_centrality(G)  
    end_time_eigenvector= time.process_time()
    elapsed_time_eigenvector = end_time_eigenvector- start_time_eigenvector
    print("elapsed_time_eigenvector", elapsed_time_eigenvector)
    print("eign")
#____________________________________________________________________________________
    start_time_clc= time.process_time()    
    clc_values = calculate_CLC_values(G, calculate_clustering_coefficient, semi_local_centralities)
    end_time_clc= time.process_time()
    elapsed_time_clc= end_time_clc- start_time_clc
    print("elapsed_time_clc", elapsed_time_clc)
    print("clc")
#________________________________________________________________________________
# Calculate pij matrix
    start_time_constraint= time.process_time()
    pij_matrix = calculate_pij(G)
    
    # Calculate constraint coefficients
    constraint_coefficients = calculate_constraint_coefficient(G, pij_matrix)
    end_time_constraint= time.process_time()
    elapsed_time_constraint= end_time_constraint- start_time_constraint
    print("elapsed_time_constraint", elapsed_time_constraint)
    print("cons")
    
#_________________________________________________________________________________
      # Calculate H-indices for each node
    start_time_hvgc= time.process_time()  
    node_h_indices = {node: calculate_node_h_index(G, node) for node in G.nodes()}
    
    # Calculate HV-indices for each node
    node_hv_indices = {node: calculate_node_hv_index(G, node, node_h_indices) for node in G.nodes()}
    
    # Calculate HVGc indices for each node
    node_hvgc_indices = {node: calculate_node_hvgc_index(G, node, node_hv_indices, constraint_coefficients, distances, R=1) for node in G.nodes()}
    end_time_hvgc= time.process_time()
    elapsed_time_hvgc= end_time_hvgc- start_time_hvgc
    print("elapsed_time_hvgc", elapsed_time_hvgc)
    print("hvgc")
#______________________________________________________________________  

#_____________________________________________________________________________________

#___________________________________________________________________
# Calculate PE for each node
    start_time_PE= time.process_time()
    PE_values = calculate_PE(G)
    end_time_PE= time.process_time()
    elapsed_time_PE= end_time_PE- start_time_PE
    print("elapsed_time_PE", elapsed_time_PE)
    print("pe")
#___________________________________________
# Calculate H values for each node
    start_time_H_values= time.process_time()
    H_values = compute_H(G)
    end_time_H_values= time.process_time()
    elapsed_time_H_values= end_time_H_values- start_time_H_values
    print("elapsed_time_H_values", elapsed_time_H_values)
    print("hello")
#_____________________________________________________
# Calculate volume centrality values for all nodes with distance h=2
    start_time_volume= time.process_time()
    volume_centrality_values = volume_centrality_for_all_nodes(G, h=2)  
    end_time_volume= time.process_time()
    elapsed_time_volume= end_time_volume- start_time_volume
    print("elapsed_time_volume", elapsed_time_volume)
    print("vol")
#______________________________________________________________________
    start_time_GIN= time.process_time()
    GIN_values = [calculate_GIN(G, node, distances, alpha=1, beta=1) for node in G.nodes()]
    end_time_GIN= time.process_time()
    elapsed_time_GIN= end_time_GIN- start_time_GIN
    print("elapsed_time_GIN", elapsed_time_GIN)
    print("gin")
    #____________________________
   # laplacian_centrality_scores = calculate_laplacian_centrality(G)
   #lapGravity
    start_time_LGC_scores= time.process_time()
    LGC_scores = calculate_LGC(G, laplacian_centrality_values, distances)
    end_time_LGC_scores= time.process_time()
    elapsed_time_LGC_scores= end_time_LGC_scores- start_time_LGC_scores
    print("elapsed_time_LGC_scores", elapsed_time_LGC_scores)
    print("lgc")
#__________________________________________________________________________
    start_time_DNC= time.process_time()
    DNC_scores = calculate_DNC(G)
    end_time_DNC= time.process_time()
    elapsed_time_DNC= end_time_DNC- start_time_DNC
    print("elapsed_time_DNC", elapsed_time_DNC)
    print("DNC")
#___________________________________________________________________
    # Calculate I_GSM_values dictionary
#    I_GSM_values = calculate_I_GSM_values(G)
    
# Calculate EIGSM values
    start_time_EIGSM= time.process_time()
    EIGSM_values = calculate_EIGSM_values(G, I_GSM_values)
    end_time_EIGSM= time.process_time()
    elapsed_time_EIGSM= end_time_EIGSM- start_time_EIGSM
    print("elapsed_time_EIGSM", elapsed_time_EIGSM)
    print("eigm")
#___________________________________________________________________________
    # Calculate societal capital scores
    start_time_societal= time.process_time()
    societal_capital_scores = calculate_societal_capital(G)
    end_time_societal= time.process_time()
    elapsed_time_societal= end_time_societal- start_time_societal
    print("elapsed_time_societal", elapsed_time_societal)
    print("sociecial")
#____________________________________
    start_time_improved_gravity= time.process_time()
    improved_gravity_index_values = {node: improved_gravity_index_centrality(G, node, distances, r=2) for node in G.nodes()}
    end_time_improved_gravity= time.process_time()
    elapsed_time_improved_gravity= end_time_improved_gravity- start_time_improved_gravity
    print("elapsed_time_improved_gravity", elapsed_time_improved_gravity)
    print("impro_gr")
#________________________________________________________________________________    
    # Calculate extended improved gravity index values for each node
    start_time_ext_improved_gravity= time.process_time()
    extended_improved_gravity_index_values = calculate_extended_improved_gravity_index_values(G, distances)
    end_time_ext_improved_gravity= time.process_time()
    elapsed_time_ext_improved_gravity= end_time_ext_improved_gravity- start_time_ext_improved_gravity
    print("elapsed_time_ext_improved_gravity", elapsed_time_ext_improved_gravity)
    print("ext_gra")
#____________________________________________________________________
   # dij_values = calculate_node_distances(G)  # Calculate node distances
    # Calculate MCGM for each node
    start_time_mcgm= time.process_time()
    mcgm_values = {}
    for node in G.nodes():
        mcgm_values[node] = calculate_mcgm(node, G, distances, r=2)
    end_time_mcgm= time.process_time()
    elapsed_time_mcgm= end_time_mcgm- start_time_mcgm
    print("elapsed_time_mcgm", elapsed_time_mcgm)
    print("mcgm")
#__________________________________________________________-
# Calculate GGC for each node
    start_time_ggc= time.process_time()    
    ggc_values = {}
    for node in G.nodes():
        ggc_values[node] = calculate_GGC(G, node, distances, R=2)
    end_time_ggc= time.process_time()
    elapsed_time_ggc= end_time_ggc- start_time_ggc
    print("elapsed_time_ggc", elapsed_time_ggc)
    print("ggc")
#_____________________________________________--

 #________________________________________________
    start_time_betweenness= time.process_time()
    
    betweenness = calculate_betweenness(G)

    # Normalize betweenness centrality for each node
          
    end_time_betweenness= time.process_time()
    elapsed_time_betweenness= end_time_betweenness- start_time_betweenness
    print("elapsed_time_betweenness", elapsed_time_betweenness)
    print("bet")
#___________________________________________________________________________
      
#____________________________________________________________________
    start_time_subgraph= time.process_time()
    subgraph_centrality= nx.subgraph_centrality(G)
    end_time_subgraph= time.process_time()
    elapsed_time_subgraph= end_time_subgraph- start_time_subgraph
    print("elapsed_time_subgraph", elapsed_time_subgraph)
    print("sub")
#_____________________________________________________________________________________
    start_time_soc= time.process_time()
    soc = nx.second_order_centrality(G) 
    end_time_soc= time.process_time()
    elapsed_time_soc= end_time_soc- start_time_soc
    print("elapsed_time_soc", elapsed_time_soc)
    print("soc")
    #________________________________________________
    # Create DataFrame
    df_attributes = pd.DataFrame({
        'Node': list(G.nodes()),
        'Degree Centrality': [centrality for _, centrality in degree_centrality.items()],
        'Betweenness Centrality': [betweenness[node] for node in G.nodes()],
        'Leverage Centrality': [leverage_centrality_values[node] for node in G.nodes()],
        'Isolating Centrality': [isolating_centrality_values[node] for node in G.nodes()], 
        'GSM': [gsm_values[node] for node in G.nodes()],  # Add GSM column
        'H-GSM_1': [H_GSM_values_1[node] for node in G.nodes()],  # Add H-GSM column
        'Local-Global Centrality': [lgc_values[node] for node in G.nodes()],  # Add LGC column
        'PageRank Centrality': [pagerank_centrality_values[node] for node in G.nodes()],  
        'Coreness Centrality': [coreness_centrality_values[node] for node in G.nodes()], 
        'Extended Coreness': [extended_coreness_values[node] for node in G.nodes()],
        'Global Relative Change in Avg Degree': [global_relative_change_avg_degree_values[node] for node in G.nodes()],
        'Global Relative Change in Avg Clustering': [global_relative_change_clustering_values[node] for node in G.nodes()], 
        'Clustering Coefficient': [clustering_coefficients_values[node] for node in G.nodes()],
        'DIL Centrality': [dil_centrality_values[node] for node in G.nodes()],
        'I-GSM Values': [I_GSM_values[node] for node in G.nodes()],
        'Clustered Local Degree': [clustered_local_degree_values[node] for node in G.nodes()],
        'Gravity Index Centrality': [gravity_index_values[node] for node in G.nodes()],
        'Gravity Model Centrality': [gravity_centrality_values[node] for node in G.nodes()],
        'Local Gravity Model Centrality': [local_gravity_centrality_values[node] for node in G.nodes()],
        'Laplacian Centrality': [laplacian_centrality_values[node] for node in G.nodes()],
        'Closeness Centrality': [closeness_centralities[node] for node in G.nodes()],
        'Semi local centrality': [semi_local_centralities[node] for node in G.nodes()],
        'Eccentricity Centrality': [eccentricity_centralities[node] for node in G.nodes()],
        'Density Centrality': [density_centralities[node] for node in G.nodes()],
        'MDD Value': [mdd_values[node] for node in G.nodes()],
        'EHCC Value': [ehcc_values[node] for node in G.nodes()],
        'HCC Value': [hcc_values[node] for node in G.nodes()],
        'P Value': [p_values[node] for node in G.nodes()],
       'Extended Degree': [extended_degrees[node] for node in G.nodes()],
        'Eigenvector Centrality': [eigenvector_centrality[node] for node in G.nodes()],
       'CLC Values': [clc_values[node] for node in G.nodes()],
        'Centrality Values': [centrality_values[node] for node in G.nodes()],
        'Constraint Coefficients': [constraint_coefficients[node] for node in G.nodes()],
        'HVGc Index': [node_hvgc_indices[node] for node in G.nodes()],
        'PE Values': [PE_values[node] for node in G.nodes()], 
        'H Values': H_values,
        'Volume Centrality': list(volume_centrality_values.values()),
        'GIN Values': GIN_values,
        'LGC Scores': LGC_scores.values(),
        'DNC Scores': DNC_scores.values(),
        'EIGSM Values': list(EIGSM_values.values()),
        'Societal Capital Scores': list(societal_capital_scores.values()),
        'Improved Gravity Index Values': list(improved_gravity_index_values.values()),
        'Extended Improved Gravity Index Values': list(extended_improved_gravity_index_values.values()),
        'Node H-Index': list(node_h_indices.values()),
        'MCGM Values': list(mcgm_values.values()),
        'GGC Values': list(ggc_values.values()),
        'Subgraph centrality': [subgraph_centrality[node] for node in G.nodes()],
        'Second order centrality': [soc[node] for node in G.nodes()],
        'K-Shell Value': list(k_shell_values.values())
    })
    
    # Save DataFrame to CSV file
  
    df_attributes.to_csv('graph_attributes_ENZYMES_g10_ext_deg_hcc_0.6_0.5hcc_check.csv', index=False)

# Call the main function
if __name__ == "__main__":
    main()
end_time_total= time.process_time()
elapsed_time_total= end_time_total- start_time_total
print("elapsed_time_total", elapsed_time_total)
