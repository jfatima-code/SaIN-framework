# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:39:34 2026

@author: Javaria
"""

import pandas as pd

# List of networks for each class
animal_social_networks = ['aves-weaver-social-16', 'aves-weaver-social-3', 'insecta-ant-colony6-c1', 
                          'mammalia-voles-bhp-trapping-c2', 'reptilia-tortoise-network-fi-c1']

social_networks = ['soc-dolphins-c1', 'soc-wiki-Vote-c1', 'socfb-Caltech36-c1', 'socfb-Reed98-c1', 'fb-pages-food-c1']

cheminformatics_networks = ['ENZYMES_g10', 'ENZYMES_g102', 'ENZYMES_g296-c1', 'NCI1-g1918-c1', 'NCI1-g2455-c1']

power_networks = ['power-1138-bus-c1', 'power-494-bus-c1', 'power-662-bus-c1', 'power-685-bus-c1', 'power-eris1176-c1']

# Path to the Excel file
file_path = "All_Networks_Features_with_majority.xlsx"

# Read the Excel file
excel_file = pd.ExcelFile(file_path)

# Function to find distinct features for each network in the class
def find_distinct_features(networks_list):
    all_features = {}
    distinct_features = {}

    # Step 1: Collect features for each network
    for network in networks_list:
        df = pd.read_excel(excel_file, sheet_name=network)
        features = set(df['selected_by_3_methods'].dropna().unique())
        all_features[network] = features

    # Step 2: Identify distinct features
    for network, features in all_features.items():
        # Combine features from all other networks in the class
        other_networks_features = set()
        for other_network in networks_list:
            if other_network != network:
                other_networks_features.update(all_features[other_network])
        
        # Find features that are unique to this network
        distinct_network_features = features - other_networks_features
        distinct_features[network] = distinct_network_features

    return distinct_features

# Function to save the distinct features for each network in the class
def save_distinct_features(class_name, networks_list):
    distinct_features = find_distinct_features(networks_list)

    # Initialize an empty DataFrame to hold all distinct features for each network in a single row
    all_distinct_features_df = pd.DataFrame()

    # Step 1: Add each network's distinct features as a new column in the DataFrame
    for network, features in distinct_features.items():
        # Convert distinct features to a single string (separated by commas)
        distinct_features_str = ', '.join(features)
        all_distinct_features_df[network] = [distinct_features_str]  # Each network's result in a single row

    # Step 2: Save the results to a new Excel file
    output_file = f"{class_name}_Distinct_Features.xlsx"
    
    # Use openpyxl as the engine for writing to Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        all_distinct_features_df.to_excel(writer, sheet_name=class_name, index=False)

# Save distinct features for each class
save_distinct_features('Animal_Social_Networks', animal_social_networks)
save_distinct_features('Social_Networks', social_networks)
save_distinct_features('Cheminformatics_Networks', cheminformatics_networks)
save_distinct_features('Power_Networks', power_networks)

print("Distinct features saved for each network in the classes.")