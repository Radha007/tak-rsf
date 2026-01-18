# vasculitis_rsf/plots.py
"""
Plotting utilities for the vasculitis_rsf toolkit.

Handles:
- Kaplan–Meier curves per leaf
- Permutation-based variable importance
- Partial dependence plots
"""

import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from sklearn.inspection import permutation_importance


# ---------------------------------------------------------
# 1. KM curves per leaf for a given tree
# ---------------------------------------------------------

def plot_km_per_leaf(
    rsf_model,
    X_all_t,
    y_struct,
    leaf_table,
    tree_index,
    out_path=None
):
    """
    Plot Kaplan–Meier curves for each leaf in a given tree.
    """

    kmf = KaplanMeierFitter()

    # Filter leaf table for this tree
    lt = leaf_table[leaf_table["Tree"] == tree_index].copy()

    # Get leaf assignments
    tree = rsf_model.estimators_[tree_index - 1]
    leaf_ids = tree.apply(X_all_t)

    plt.figure(figsize=(10, 7))

    for _, row in lt.iterrows():
        leaf_id = row["Leaf_ID"]
        idx = np.where(leaf_ids == leaf_id)[0]

        # Skip tiny leaves
        if len(idx) < 3:
            continue

        times = y_struct["time"][idx]
        events = y_struct["event"][idx]

        label = f"Leaf {leaf_id} (n={len(idx)}, {row['Hazard_Direction']})"
        kmf.fit(times, events, label=label)
        kmf.plot(ci_show=False)

    plt.title(f"Kaplan–Meier Curves per Leaf — Tree {tree_index}")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.legend(fontsize=8)
    plt.tight_layout()

    if out_path is None:
        out_path = f"KM_per_leaf_Tree{tree_index}.png"

    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OUTPUT] Saved {out_path}")
    return out_path


# ---------------------------------------------------------
# 2. Permutation-based variable importance
# ---------------------------------------------------------

ddef plot_variable_importance(
    rsf_model,
    X_test_t,
    y_test,
    feature_names,
    top_n=20,
    out_path="RSF_variable_importance.png"
):
    """
    Manual permutation importance for scikit-survival RSF.
    """

    # Baseline risk = negative survival at last time point
    surv_funcs = rsf_model.predict_survival_function(X_test_t)
    baseline_risk = np.array([-sf.y[-1] for sf in surv_funcs], dtype=float)
    baseline_score = baseline_risk.mean()

    importances = []

    X_copy = X_test_t.copy()

    for j in range(X_copy.shape[1]):
        X_perm = X_copy.copy()
        np.random.shuffle(X_perm[:, j])

        surv_perm = rsf_model.predict_survival_function(X_perm)
        perm_risk = np.array([-sf.y[-1] for sf in surv_perm], dtype=float)
        perm_score = perm_risk.mean()

        importance = perm_score - baseline_score
        importances.append(importance)

    importances = np.array(importances)
    order = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[order][::-1], importances[order][::-1])
    plt.xlabel("Permutation Importance (Increase in Mean Risk)")
    plt.title("Random Survival Forest — Variable Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OUTPUT] Saved {out_path}")
    return out_path


# ---------------------------------------------------------
# 3. Partial dependence plots
# ---------------------------------------------------------

def partial_dependence_risk(
    rsf_model,
    X_all_t,
    feature_names,
    var_name,
    y_struct,
    grid_points=20
):
    """
    Compute partial dependence for a single variable.
    """

    surv_funcs = rsf_model.predict_survival_function(X_all_t[:5])
    time_grid = surv_funcs[0].x

    # Time horizon = median observed time
    time_horizon = np.median(y_struct["time"])
    t_idx = np.argmin(np.abs(time_grid - time_horizon))

    # Find columns matching var_name (handles one-hot)
    cols = [i for i, f in enumerate(feature_names) if f.startswith(var_name)]
    if not cols:
        print(f"[WARN] No columns found for variable '{var_name}'")
        return None

    X_ref = X_all_t.copy()

    values = np.linspace(
        np.percentile(X_ref[:, cols].ravel(), 5),
        np.percentile(X_ref[:, cols].ravel(), 95),
        grid_points
    )

    risks = []

    for v in values:
        X_mod = X_ref.copy()
        for c in cols:
            X_mod[:, c] = v

        surv_mod = rsf_model.predict_survival_function(X_mod)
        risk_vals = np.array([1.0 - sf.y[t_idx] for sf in surv_mod])
        risks.append(risk_vals.mean())

    return values, np.array(risks), time_horizon


def plot_partial_dependence(
    rsf_model,
    X_all_t,
    feature_names,
    y_struct,
    var_names,
    out_path="RSF_partial_dependence.png"
):
    """
    Plot partial dependence curves for multiple variables.
    """

    plt.figure(figsize=(10, 6))

    for var in var_names:
        res = partial_dependence_risk(
            rsf_model,
            X_all_t,
            feature_names,
            var,
            y_struct
        )
        if res is None:
            continue

        values, risks, t_h = res
        plt.plot(values, risks, label=f"{var} (t={t_h:.1f})")

    plt.xlabel("Value")
    plt.ylabel("Mean Predicted Risk (1 - S(t))")
    plt.title("Partial Dependence of Predicted Risk")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OUTPUT] Saved {out_path}")
    return out_path