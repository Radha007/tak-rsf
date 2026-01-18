import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Any

# Import the same helpers your CLI uses
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

# Cache data and model so the app is responsive
@st.cache_data
def cached_load_data():
    return load_data()

@st.cache_resource
def cached_fit_rsf(X_train_t, y_train):
    return fit_rsf(X_train_t, y_train)

def show_plot_or_image(obj: Any):
    """
    Helper: display whatever the plotting function returns.
    If it returns a file path (str), show image. If it returns a Matplotlib figure, show it.
    """
    if obj is None:
        st.info("No return value from plot function. Check if it saved a file instead.")
        return
    if isinstance(obj, str) and os.path.exists(obj):
        st.image(obj)
        return
    # matplotlib figure
    try:
        st.pyplot(obj)
        return
    except Exception:
        pass
    # pandas dataframe
    if isinstance(obj, pd.DataFrame):
        st.dataframe(obj)
        return
    st.write(obj)

def main():
    st.title("Vasculitis RSF — Streamlit UI")

    # Load data
    with st.spinner("Loading data..."):
        (
            X_train_t,
            X_test_t,
            X_all_t,
            y_train,
            y_test,
            feature_names,
            preprocessor,
            X_raw,
            y_struct,
        ) = cached_load_data()

    st.sidebar.header("Actions")
    do_fit = st.sidebar.button("Fit RSF model")
    show_cindex = st.sidebar.button("Compute C-index")
    tree_idx = st.sidebar.number_input("Tree index (for rules / KM)", min_value=1, value=1, step=1)
    do_tree = st.sidebar.button("Show tree rules")
    do_km = st.sidebar.button("Show KM per leaf")
    do_vi = st.sidebar.button("Variable importance")
    pd_vars = st.sidebar.multiselect("Partial dependence vars", feature_names, default=feature_names[:3])
    do_pd = st.sidebar.button("Show partial dependence")
    do_report = st.sidebar.button("Build PDF report")

    # Fit model only when requested
    rsf_model = None
    if do_fit:
        with st.spinner("Fitting RSF (may take a while)..."):
            rsf_model = cached_fit_rsf(X_train_t, y_train)
        st.success("Model fitted")
    else:
        st.info("Press 'Fit RSF model' to train the model (stored in cache)")

    # If not fitted this session but cached exists, try to get it
    if rsf_model is None:
        try:
            # cached_fit_rsf will return existing resource if already created
            rsf_model = cached_fit_rsf(X_train_t, y_train)
        except Exception:
            pass

    if show_cindex:
        if rsf_model is None:
            st.error("Model not fitted. Please fit first.")
        else:
            c_index = compute_c_index(rsf_model, X_test_t, y_test)
            st.write(f"C-index: {c_index:.3f}")

    if do_tree:
        if rsf_model is None:
            st.error("Model not fitted. Please fit first.")
        else:
            # pretty_print_tree_rules likely prints to stdout; capture/convert to text if possible
            try:
                text = pretty_print_tree_rules(rsf_model.estimators_[tree_idx - 1], feature_names, tree_index=tree_idx)
                # If function prints rather than return, you might adapt it to return string.
                if text:
                    st.text(text)
                else:
                    st.info("pretty_print_tree_rules returned nothing — check function or adapt it to return text.")
            except Exception as e:
                st.error(f"Error showing tree rules: {e}")

    if do_km:
        if rsf_model is None:
            st.error("Model not fitted. Please fit first.")
        else:
            leaf_table = compute_leaf_survival_summary(rsf_model, X_all_t, y_struct, feature_names)
            km_obj = plot_km_per_leaf(rsf_model, X_all_t, y_struct, leaf_table, tree_index=tree_idx)
            show_plot_or_image(km_obj)

    if do_vi:
        if rsf_model is None:
            st.error("Model not fitted. Please fit first.")
        else:
            vi_obj = plot_variable_importance(rsf_model, X_test_t, y_test, feature_names)
            show_plot_or_image(vi_obj)

    if do_pd:
        if rsf_model is None:
            st.error("Model not fitted. Please fit first.")
        else:
            pd_obj = plot_partial_dependence(
                rsf_model, X_all_t, feature_names, y_struct, var_names=pd_vars
            )
            show_plot_or_image(pd_obj)

    if do_report:
        if rsf_model is None:
            st.error("Model not fitted. Please fit first.")
        else:
            st.info("Exporting rules, leaf table and generating figures for PDF...")
            leaf_table = compute_leaf_survival_summary(rsf_model, X_all_t, y_struct, feature_names)
            df_rules = export_tree_rules(rsf_model, feature_names)
            tree1_text = df_rules[df_rules["Tree"] == 1].to_string(index=False)
            tree2_text = df_rules[df_rules["Tree"] == 2].to_string(index=False)
            export_leaf_table(leaf_table)
            km1 = plot_km_per_leaf(rsf_model, X_all_t, y_struct, leaf_table, 1)
            km2 = plot_km_per_leaf(rsf_model, X_all_t, y_struct, leaf_table, 2)
            vi = plot_variable_importance(rsf_model, X_test_t, y_test, feature_names)
            pd_path = plot_partial_dependence(
                rsf_model, X_all_t, feature_names, y_struct, var_names=pd_vars or ["Age", "ESR", "CRP"]
            )
            out = build_pdf_report(
                tree1_rules_text=tree1_text,
                tree2_rules_text=tree2_text,
                leaf_table_df=leaf_table,
                km_tree1_path=km1,
                km_tree2_path=km2,
                vi_path=vi,
                pd_path=pd_path,
            )
            st.success("Report built")
            if isinstance(out, str) and os.path.exists(out):
                st.download_button("Download report", open(out, "rb").read(), file_name=os.path.basename(out))

if __name__ == "__main__":
    main()