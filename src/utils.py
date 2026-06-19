"""
Functions for the multilevel mixed-effect modeling pipeline.
"""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


# fit_mixedlm
def fit_mixedlm(formula, data, groups, re_formula=None):
    if re_formula is None:
        re_formula = "~1"

    methods = [
        ({"reml": True}, "converged_default"),
        ({"reml": True, "method": "lbfgs", "maxiter": 500}, "converged_lbfgs"),
        ({"reml": True, "method": "nm", "maxiter": 1000}, "converged_nm"),
        ({"reml": True, "method": "powell", "maxiter": 1000}, "converged_powell"),
    ]

    try:
        model = smf.mixedlm(formula, data=data, groups=groups, re_formula=re_formula)
    except Exception as e:
        return None, f"failed: {str(e)}"

    for kwargs, status in methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = model.fit(**kwargs)
            if getattr(fit, "converged", False):
                return fit, status
        except Exception:
            continue

    try:
        fallback_model = smf.mixedlm(formula, data=data, groups=groups, re_formula="~1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fallback_model.fit(reml=True)
        return fit, "fallback_intercept_only"
    except Exception as e:
        return None, f"failed: {str(e)}"


def extract_variance_components(fit, has_random_slope=False):
    """
    Extract (var_u0, var_u1, cov_u01, cor_u01, var_e) from a fitted model.
    var_u1 / cov_u01 / cor_u01 are 0 when there is no random slope.
    """
    G = fit.cov_re
    var_u0 = float(G.iloc[0, 0])
    var_u1, cov_u01, cor_u01 = 0.0, 0.0, 0.0
    if has_random_slope and G.shape[0] >= 2:
        var_u1 = float(G.iloc[1, 1])
        cov_u01 = float(G.iloc[0, 1])
        if var_u0 > 0 and var_u1 > 0:
            cor_u01 = cov_u01 / np.sqrt(var_u0 * var_u1)
    var_e = fit.scale
    return var_u0, var_u1, cov_u01, cor_u01, var_e


# likelihood ratio test with boundary correction
def lrt_boundary_corrected(llf_full, llf_ref, n_extra=2):
    """
    LRT with the boundary-corrected chi-square mixtures.

    n_extra=2: one random slope added to model
      0.5 * chi2(1) + 0.5 * chi2(2)
    n_extra=5: two random slopes added to model
      1/3 * chi2(3) + 1/3 * chi2(4) + 1/3 * chi2(5)
    """
    lrt_stat = max(2 * (llf_full - llf_ref), 0.0)
    if n_extra == 2:
        p_value = 0.5 * chi2.sf(lrt_stat, 1) + 0.5 * chi2.sf(lrt_stat, 2)
    elif n_extra == 5:
        p_value = (
            (1 / 3) * chi2.sf(lrt_stat, 3)
            + (1 / 3) * chi2.sf(lrt_stat, 4)
            + (1 / 3) * chi2.sf(lrt_stat, 5)
        )
    else:
        p_value = chi2.sf(lrt_stat, n_extra)
    return lrt_stat, p_value

