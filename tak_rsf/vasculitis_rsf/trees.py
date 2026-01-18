# vasculitis_rsf/trees.py
"""
Tree rule extraction utilities for the vasculitis_rsf toolkit.

Handles:
- Extracting root→leaf rule paths from RSF trees
- Pretty-printing Tree 1 and Tree 2
- Exporting rule tables
"""

import pandas as pd
from sklearn.tree import _tree


# ---------------------------------------------------------
# 1. Extract rules from a single tree
# ---------------------------------------------------------

def extract_rules_from_tree(tree, feature_names):
    """
    Extract all root→leaf rule paths from a single survival tree.
    Returns a list of dicts:
        {
            "leaf_id": int,
            "rule": "Feature ≤ threshold & ...",
            "node_samples": int
        }
    """

    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    children_left = tree_.children_left
    children_right = tree_.children_right
    n_node_samples = tree_.n_node_samples

    rules = []

    def recurse(node_id, conditions):
        # Leaf node
        if children_left[node_id] == _tree.TREE_LEAF:
            rules.append({
                "leaf_id": node_id,
                "rule": " & ".join(conditions) if conditions else "ALL",
                "node_samples": int(n_node_samples[node_id]),
            })
        else:
            feat = feature_names[feature[node_id]]
            thr = threshold[node_id]

            # Left branch
            recurse(
                children_left[node_id],
                conditions + [f"{feat} ≤ {thr:.3f}"]
            )

            # Right branch
            recurse(
                children_right[node_id],
                conditions + [f"{feat} > {thr:.3f}"]
            )

    recurse(0, [])
    return rules


# ---------------------------------------------------------
# 2. Pretty-print a tree's rules
# ---------------------------------------------------------

def pretty_print_tree_rules(tree, feature_names, tree_index=1):
    """
    Print a clean, human-readable rule diagram for a given tree.
    """

    rules = extract_rules_from_tree(tree, feature_names)

    print("\n" + "=" * 60)
    print(f"                 RSF TREE {tree_index} — RULE DIAGRAM")
    print("=" * 60)

    for r in rules:
        print(
            f"Leaf {r['leaf_id']:3d} | "
            f"n={r['node_samples']:3d} | "
            f"{r['rule']}"
        )

    print("=" * 60 + "\n")


# ---------------------------------------------------------
# 3. Export rules for Tree 1 and Tree 2
# ---------------------------------------------------------

def export_tree_rules(rsf_model, feature_names, out_path="RSF_Tree1_Tree2_rules.csv"):
    """
    Export rule tables for Tree 1 and Tree 2 into a CSV file.
    """

    tree1_rules = extract_rules_from_tree(rsf_model.estimators_[0], feature_names)
    tree2_rules = extract_rules_from_tree(rsf_model.estimators_[1], feature_names)

    df1 = pd.DataFrame(tree1_rules)
    df1["Tree"] = 1

    df2 = pd.DataFrame(tree2_rules)
    df2["Tree"] = 2

    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv(out_path, index=False)

    print(f"[OUTPUT] Saved {out_path}")
    return df