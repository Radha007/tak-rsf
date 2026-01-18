# vasculitis_rsf/leaves.py
"""
Leaf-level survival analysis for the vasculitis_rsf toolkit.

Handles:
- Assigning samples to leaves for each tree
- Computing survival at a chosen time horizon
- Computing hazard direction (↑ high risk, ↓ low risk, ↔ intermediate)
- Building a manuscript-ready leaf summary table
"""

import numpy as np
import pandas as pd
from .trees import extract_rules_from_tree


# ---------------------------------------------------------
# 1. Compute leaf-level survival summary
# ---------------------------------------------------------

def compute_leaf_survival_summary(
    rsf_model,
    X_all_t,
    y_struct,
    feature_names,
    time_horizon=None
):
    """
    Compute leaf-level survival summary for all trees in the RSF.

    Returns a DataFrame with:
        - Tree index
        - Leaf ID
        - Rule path
        - Node sample counts
        - Mean predicted survival at time_horizon
        - Mean observed time
        - Event rate
        - Hazard direction
    """

    # Predict survival functions for all samples
    surv_funcs = rsf_model.predict_survival_function(X_all_t)

    # Time grid from first survival function
    time_grid = surv_funcs[0].x

    # Default time horizon = median observed time
    if time_horizon is None:
        time_horizon = np.median(y_struct["time"])

    # Find nearest time index
    t_idx = np.argmin(np.abs(time_grid - time_horizon))

    # Overall survival at time_horizon
    overall_surv = np.mean([sf.y[t_idx] for sf in surv_funcs])

    rows = []

    # Loop through each tree in the forest
    for t_idx_tree, tree in enumerate(rsf_model.estimators_):

        # Assign each sample to a leaf
        leaf_ids = tree.apply(X_all_t)

        # Extract rule paths
        rules = extract_rules_from_tree(tree, feature_names)
        rule_map = {r["leaf_id"]: r for r in rules}

        # For each leaf in this tree
        for leaf in np.unique(leaf_ids):

            idx = np.where(leaf_ids == leaf)[0]

            # Predicted survival at time_horizon
            surv_at_t = np.mean([surv_funcs[i].y[t_idx] for i in idx])

            # Observed outcomes
            mean_time = float(np.mean(y_struct["time"][idx]))
            event_rate = float(np.mean(y_struct["event"][idx]))

            # Hazard direction
            diff = surv_at_t - overall_surv
            if diff <= -0.05:
                hazard = "↑ high risk"
            elif diff >= 0.05:
                hazard = "↓ low risk"
            else:
                hazard = "↔ intermediate"

            r = rule_map[leaf]

            rows.append({
                "Tree": t_idx_tree + 1,
                "Leaf_ID": int(leaf),
                "Rule_Path": r["rule"],
                "Node_Samples_Model": r["node_samples"],
                "Node_Samples_Actual": len(idx),
                f"Mean_Pred_Survival_at_{time_horizon:.1f}": float(surv_at_t),
                "Mean_Observed_Time": mean_time,
                "Event_Rate": event_rate,
                "Hazard_Direction": hazard,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["Tree", "Event_Rate"], ascending=[True, False])
    return df


# ---------------------------------------------------------
# 2. Export leaf table to CSV
# ---------------------------------------------------------

def export_leaf_table(df, out_path="RSF_leaf_rules_table.csv"):
    """
    Save the leaf-level survival summary table to CSV.
    """
    df.to_csv(out_path, index=False)
    print(f"[OUTPUT] Saved {out_path}")
    return df