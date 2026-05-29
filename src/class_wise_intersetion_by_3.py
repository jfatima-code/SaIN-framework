# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 08:26:27 2026

@author: Javaria
"""
import pandas as pd
from collections import defaultdict

# ================= PATHS =================
MAJORITY_FILE = r".....\All_Networks_Features_with_majority.xlsx"
NETWORK_CLASS_FILE = r".....\network_classes.xlsx"

OUTPUT_FILE = r".........\Class_Wise_Intersection_SelectedBy3.xlsx"
# ========================================


# ---------- Read network-class file (NO HEADER) ----------
net_cls_df = pd.read_excel(
    NETWORK_CLASS_FILE,
    header=None
)

# First column = Network, Second column = Class
net_cls_df.columns = ["Network", "Class"]

net_cls_df["Network"] = net_cls_df["Network"].astype(str).str.strip()
net_cls_df["Class"] = net_cls_df["Class"].astype(str).str.strip()

# network → class mapping
network_to_class = dict(
    zip(net_cls_df["Network"], net_cls_df["Class"])
)

# ---------- Read majority file ----------
xls = pd.ExcelFile(MAJORITY_FILE)

# class → list of attribute sets (one per network)
class_to_sets = defaultdict(list)

# ---------- Collect selected_by_3_methods ----------
for network in xls.sheet_names:

    # skip networks not listed in class file
    if network not in network_to_class:
        continue

    df = pd.read_excel(MAJORITY_FILE, sheet_name=network)

    if "selected_by_3_methods" not in df.columns:
        continue

    attrs = (
        df["selected_by_3_methods"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    if len(attrs) == 0:
        continue

    cls = network_to_class[network]
    class_to_sets[cls].append(set(attrs))

# ---------- Write separate sheet per network class ----------
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    for cls, sets_list in class_to_sets.items():

        # intersection across networks of same class
        if len(sets_list) >= 2:
            inter = set.intersection(*sets_list)
        else:
            inter = sets_list[0]

        inter = sorted(inter)

        out_df = pd.DataFrame(
            {"selected_by_3_methods": inter}
        )

        sheet_name = cls[:31].replace(" ", "_")

        out_df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )

print("🎉 DONE!")
print("📁 File saved at:", OUTPUT_FILE)

