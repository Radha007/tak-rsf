# cli.py
"""
Command-line interface for the vasculitis_rsf toolkit.

Usage examples:
    python cli.py --fit
    python cli.py --tree 1
    python cli.py --km 2
    python cli.py --vi
    python cli.py --pd Age ESR CRP
    python cli.py --report
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import argparse
import pandas as pd

from vasculitis_rsf.data import load_data
from vasculitis_rsf.model import fit_rsf, compute_c_index
from vasculitis_rsf.trees import pretty_print_tree_rules, export_tree_rules
from vasculitis_rsf.leaves import compute_leaf_survival_summary, export_leaf_table
from vasculitis_rsf.plots import (
    plot_km_per_leaf,
    plot_variable_importance,
    plot_partial_dependence,
)
from vasculitis_rsf.report import build_pdf_report


# ---------------------------------------------------------
# 1. Parse CLI arguments
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Vasculitis RSF Toolkit")

    parser.add_argument("--fit", action="store_true",
                        help="Fit RSF model and compute C-index")

    parser.add_argument("--tree", type=int,
                        help="Print rule diagram for Tree N")

    parser.add_argument("--km", type=int,
                        help="Generate KM curves for Tree N")

    parser.add_argument("--vi", action="store_true",
                        help="Generate variable importance plot")

    parser.add_argument("--pd", nargs="+",
                        help="Generate partial dependence for variables")

    parser.add_argument("--report", action="store_true",
                        help="Generate full PDF report")

    return parser.parse_args()


# ---------------------------------------------------------
# 2. Main CLI logic
# ---------------------------------------------------------

def main():
    args = parse_args()

    # Load data
    (
        X_train_t,
        X_test_t,
        X_all_t,
        y_train,
        y_test,
        feature_names,
        preprocessor,
        X_raw,
        y_struct
    ) = load_data()

    # Fit RSF
    rsf_model = fit_rsf(X_train_t, y_train)

    # Compute leaf table (used by many commands)
    leaf_table = compute_leaf_survival_summary(
        rsf_model, X_all_t, y_struct, feature_names
    )

    # -----------------------------------------------------
    # --fit
    # -----------------------------------------------------
    if args.fit:
        c_index = compute_c_index(rsf_model, X_test_t, y_test)
        print(f"[RESULT] C-index: {c_index:.3f}")
        return

    # -----------------------------------------------------
    # --tree N
    # -----------------------------------------------------
    if args.tree:
        pretty_print_tree_rules(
            rsf_model.estimators_[args.tree - 1],
            feature_names,
            tree_index=args.tree
        )
        return

    # -----------------------------------------------------
    # --km N
    # -----------------------------------------------------
    if args.km:
        plot_km_per_leaf(
            rsf_model,
            X_all_t,
            y_struct,
            leaf_table,
            tree_index=args.km
        )
        return

    # -----------------------------------------------------
    # --vi
    # -----------------------------------------------------
    if args.vi:
        plot_variable_importance(
            rsf_model,
            X_test_t,
            y_test,
            feature_names
        )
        return

    # -----------------------------------------------------
    # --pd var1 var2 ...
    # -----------------------------------------------------
    if args.pd:
        plot_partial_dependence(
            rsf_model,
            X_all_t,
            feature_names,
            y_struct,
            var_names=args.pd
        )
        return

    # -----------------------------------------------------
    # --report
    # -----------------------------------------------------
    if args.report:

        # Export tree rules
        df_rules = export_tree_rules(rsf_model, feature_names)

        # Convert rules to text
        tree1_text = df_rules[df_rules["Tree"] == 1].to_string(index=False)
        tree2_text = df_rules[df_rules["Tree"] == 2].to_string(index=False)

        # Export leaf table
        export_leaf_table(leaf_table)

        # Generate figures
        km1 = plot_km_per_leaf(rsf_model, X_all_t, y_struct, leaf_table, 1)
        km2 = plot_km_per_leaf(rsf_model, X_all_t, y_struct, leaf_table, 2)
        vi = plot_variable_importance(rsf_model, X_test_t, y_test, feature_names)
        pd = plot_partial_dependence(
            rsf_model, X_all_t, feature_names, y_struct,
            var_names=["Age", "ESR", "CRP"]
        )

        # Build PDF
        build_pdf_report(
            tree1_rules_text=tree1_text,
            tree2_rules_text=tree2_text,
            leaf_table_df=leaf_table,
            km_tree1_path=km1,
            km_tree2_path=km2,
            vi_path=vi,
            pd_path=pd
        )
        return


if __name__ == "__main__":
    main()