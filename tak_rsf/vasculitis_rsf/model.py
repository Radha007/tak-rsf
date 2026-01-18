# vasculitis_rsf/model.py
"""
Model utilities for the vasculitis_rsf toolkit.

Handles:
- Fitting Random Survival Forest (RSF)
- Computing C-index
- Predicting survival functions
- Risk scoring
"""

import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored


# ---------------------------------------------------------
# 1. Fit RSF model
# ---------------------------------------------------------

def fit_rsf(
    X_train_t,
    y_train,
    n_estimators=50,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
):
    """
    Fit a Random Survival Forest (RSF) model.
    Returns the fitted model.
    """

    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    rsf.fit(X_train_t, y_train)
    return rsf


# ---------------------------------------------------------
# 2. Compute C-index
# ---------------------------------------------------------

def compute_c_index(rsf_model, X_test_t, y_test):
    """
    Compute C-index using negative survival at last time point as risk score.
    """

    surv_funcs = rsf_model.predict_survival_function(X_test_t)
    risks = np.array([-sf.y[-1] for sf in surv_funcs], dtype=float)

    result = concordance_index_censored(
        y_test["event"].astype(bool),
        y_test["time"].astype(float),
        risks
    )
    return result[0]  # c-index


# ---------------------------------------------------------
# 3. Predict survival functions
# ---------------------------------------------------------

def predict_survival(rsf_model, X_t):
    """
    Predict survival functions for transformed data.
    Returns a list of StepFunction objects.
    """
    return rsf_model.predict_survival_function(X_t)


# ---------------------------------------------------------
# 4. Compute risk scores
# ---------------------------------------------------------

def compute_risk_scores(rsf_model, X_t):
    """
    Compute risk scores as negative survival at last time point.
    Higher = higher risk.
    """
    surv_funcs = rsf_model.predict_survival_function(X_t)
    risks = np.array([-sf.y[-1] for sf in surv_funcs], dtype=float)
    return risks