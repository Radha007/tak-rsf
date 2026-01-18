# rsf_module/data.py
"""
Data loading and preprocessing for the vasculitis_rsf toolkit.
Handles:
- Loading TAK dataset
- Selecting predictors (Option B: includes treatment variables)
- Preprocessing (numeric + one-hot categorical)
- float32 conversion for scikit-survival compatibility
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sksurv.util import Surv


# ---------------------------------------------------------
# 1. Feature specification (Option B: includes treatment)
# ---------------------------------------------------------

CORE_FEATURES = ["Age", "Sex", "ESR", "CRP"]

ORGAN_FEATURES = [
    "DAH", "RPGN", "Severe_Neuropathy", "GI_Ischemia", "Cardiac_Involvement",
    "GN", "Orbital", "Subglottic", "Pulmonary_Nodules", "Scleritis",
    "Skin_Ulcers", "ENT_Limited", "Arthritis", "Skin_Purpura",
    "Constitutional",
]

DISEASE_FEATURES = [
    "ANCA_Type", "eGFR", "BVAS", "Severity",
    "Organ_Threat_Tight", "RTX_Favored_PR3",
    "Disease_Duration_months",
]

TREATMENT_FEATURES = ["Induction_Plan", "Trt-1", "Trt-2"]


# ---------------------------------------------------------
# 2. Build preprocessing pipeline
# ---------------------------------------------------------

def build_preprocessor(X):
    """Create a ColumnTransformer for numeric + categorical preprocessing."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def get_feature_names(preprocessor, numeric_cols, categorical_cols):
    """Return expanded feature names after one-hot encoding."""
    feature_names = list(numeric_cols)
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat":
            ohe = trans
            feature_names.extend(ohe.get_feature_names_out(categorical_cols))
    return feature_names


# ---------------------------------------------------------
# 3. Main loader
# ---------------------------------------------------------

def load_data(
    data_path="TAK-Data.csv",
    test_size=0.2,
    random_state=42
):
    """
    Load TAK dataset, preprocess, and return:
    - X_train_t, X_test_t, X_all_t (float32)
    - y_train, y_test
    - feature_names
    - preprocessor (fitted)
    """

    df = pd.read_csv(data_path)

    # Survival outcome
    y = Surv.from_arrays(
        event=df["Event"].astype(bool),
        time=df["Observed_Time"].astype(float),
    )

    # Select predictors
    feature_cols = (
        CORE_FEATURES +
        ORGAN_FEATURES +
        DISEASE_FEATURES +
        TREATMENT_FEATURES
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()

    # Build preprocessing
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    # Fit-transform split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Fit preprocessor on training data only
    preprocessor.fit(X_train)

    # Transform all sets
    X_train_t = preprocessor.transform(X_train).astype(np.float32)
    X_test_t = preprocessor.transform(X_test).astype(np.float32)
    X_all_t = preprocessor.transform(X).astype(np.float32)

    # Expanded feature names
    feature_names = get_feature_names(preprocessor, numeric_cols, categorical_cols)

    return (
        X_train_t,
        X_test_t,
        X_all_t,
        y_train,
        y_test,
        feature_names,
        preprocessor,
        X,
        y
    )