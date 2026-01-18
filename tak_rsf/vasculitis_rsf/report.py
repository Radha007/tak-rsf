# vasculitis_rsf/report.py
"""
PDF report generator for the vasculitis_rsf toolkit.

Combines:
- Tree 1 and Tree 2 rule diagrams
- KM curves per leaf
- Variable importance plot
- Partial dependence plot
- Leaf table summary

into a single multi-page PDF.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


# ---------------------------------------------------------
# 1. Add a text page (for rules, tables, summaries)
# ---------------------------------------------------------

def _add_text_page(pdf, title, text):
    """
    Add a full-page text block to the PDF.
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()

    plt.text(
        0.01, 0.99,
        f"{title}\n\n{text}",
        va="top",
        ha="left",
        fontsize=10,
        family="monospace"
    )

    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------
# 2. Add an image page (for PNG figures)
# ---------------------------------------------------------

def _add_image_page(pdf, image_path, title=None):
    """
    Add a PNG image as a full page in the PDF.
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()

    if title:
        plt.title(title, fontsize=14)

    img = plt.imread(image_path)
    plt.imshow(img)
    plt.axis("off")

    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------
# 3. Build the full PDF report
# ---------------------------------------------------------

def build_pdf_report(
    out_path="RSF_Report.pdf",
    tree1_rules_text=None,
    tree2_rules_text=None,
    leaf_table_df=None,
    km_tree1_path=None,
    km_tree2_path=None,
    vi_path=None,
    pd_path=None
):
    """
    Create a multi-page PDF report combining:
    - Tree 1 rules
    - Tree 2 rules
    - Leaf table summary
    - KM curves
    - Variable importance
    - Partial dependence
    """

    with PdfPages(out_path) as pdf:

        # Tree 1 rules
        if tree1_rules_text:
            _add_text_page(pdf, "RSF TREE 1 — RULE DIAGRAM", tree1_rules_text)

        # Tree 2 rules
        if tree2_rules_text:
            _add_text_page(pdf, "RSF TREE 2 — RULE DIAGRAM", tree2_rules_text)

        # Leaf table summary
        if leaf_table_df is not None:
            table_text = leaf_table_df.to_string(index=False)
            _add_text_page(pdf, "LEAF-LEVEL SURVIVAL SUMMARY", table_text)

        # KM curves
        if km_tree1_path:
            _add_image_page(pdf, km_tree1_path, "KM Curves — Tree 1")

        if km_tree2_path:
            _add_image_page(pdf, km_tree2_path, "KM Curves — Tree 2")

        # Variable importance
        if vi_path:
            _add_image_page(pdf, vi_path, "Variable Importance")

        # Partial dependence
        if pd_path:
            _add_image_page(pdf, pd_path, "Partial Dependence")

    print(f"[OUTPUT] Saved PDF report: {out_path}")
    return out_path