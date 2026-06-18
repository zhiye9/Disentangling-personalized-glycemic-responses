"""
Progressive multilevel model building for each response variable, from null model to nutrients and meal timing, and add random slope to carb. Extract variance components, compute ICC, and perform LRT for random slope.  Then extract BLUPs and compute reliability and meals needed for target reliability levels.

This script should be run after feature_extraction_preprocessing.py.
"""

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy import stats

from utils import (
    fit_mixedlm,
    extract_variance_components,
    lrt_boundary_corrected,
)

MIN_MEALS = 5

# configuration for formulas
progressive_results = {}
stage3_baseline = {}

base_cols_nutrients = [
    "carb_within", "carb_between",
    PROTEIN_COL, FAT_COL, FIBER_COL, "has_alcohol",
]

FORMULA_NUTRIENTS_BASE = (
    f"carb_within + carb_between + {PROTEIN_COL} + {FAT_COL} + {FIBER_COL} + has_alcohol"
)
FORMULA_NUTRIENTS = f"{FORMULA_NUTRIENTS_BASE} + C({MEAL_CATEGORY_COL})"
base_cols_full = base_cols_nutrients + [MEAL_CATEGORY_COL]

# progressive modeling for each response variable
for idx, (resp_col, resp_name, is_log) in enumerate(response_runs):
    tag = "[LOG]" if is_log else "[RAW]"

    progressive_results[resp_name] = {}

    df_resp = df_model.dropna(subset=[resp_col] + base_cols_full).copy()
    mc = df_resp.groupby("participant_id").size()
    df_resp = df_resp[df_resp["participant_id"].isin(
        mc[mc >= MIN_MEALS].index)].copy()

    n_obs = len(df_resp)
    n_groups = df_resp["participant_id"].nunique()

    # stage 1: null model
    fit_s1, st_s1 = fit_mixedlm(
        f"{resp_col} ~ 1", df_resp,
        groups=df_resp["participant_id"], re_formula="~1")
    print(f"[{st_s1}]")
    if fit_s1 is None:
        continue
    var_u0_s1 = float(fit_s1.cov_re.iloc[0, 0])
    var_e_s1 = fit_s1.scale
    icc_null = var_u0_s1 / (var_u0_s1 + var_e_s1)
    progressive_results[resp_name]["Stage1_null"] = {
        "var_u0": var_u0_s1, "var_e": var_e_s1, "icc": icc_null,
        "aic": fit_s1.aic, "n_obs": n_obs, "n_groups": n_groups,
    }
    print(f"    ICC_null = {icc_null:.4f} ({100 * icc_null:.1f}% between-person)")

    # Stage 2a: nutrients 
    fit_s2a, st_s2a = fit_mixedlm(
        f"{resp_col} ~ {FORMULA_NUTRIENTS_BASE}", df_resp,
        groups=df_resp["participant_id"], re_formula="~1")
    print(f"[{st_s2a}]")

    var_u0_s2a, _, _, _, var_e_s2a = extract_variance_components(fit_s2a)
    delta_e_s2a = 100 * (1 - var_e_s2a / var_e_s1)
    progressive_results[resp_name]["Stage2a_nutrients"] = {
        "var_u0": var_u0_s2a, "var_e": var_e_s2a,
        "icc": var_u0_s2a / (var_u0_s2a + var_e_s2a),
        "aic": fit_s2a.aic, "delta_var_e_pct": delta_e_s2a,
    }
    print(f"    var_e reduction: {delta_e_s2a:.1f}%")

    # stage 2b: meal timing
    fit_s2b, st_s2b = fit_mixedlm(
        f"{resp_col} ~ {FORMULA_NUTRIENTS}", df_resp,
        groups=df_resp["participant_id"], re_formula="~1")
    print(f"[{st_s2b}]")

    # stage 3: random carb slope
    fit_s3, st_s3 = fit_mixedlm(
        f"{resp_col} ~ {FORMULA_NUTRIENTS}", df_resp,
        groups=df_resp["participant_id"], re_formula=f"~{CARB_SLOPE_VAR}")
    print(f"[{st_s3}]")

    var_u0_s3, var_u1_s3, cov_u01_s3, cor_u01_s3, var_e_s3 = \
        extract_variance_components(fit_s3, has_random_slope=True)
    icc_s3 = (
        var_u0_s3 / (var_u0_s3 + var_e_s3)
        if (var_u0_s3 + var_e_s3) > 0 else 0
    )
    lrt_stat, lrt_p = lrt_boundary_corrected(fit_s3.llf, fit_s2b.llf, n_extra=2)

    # R2 decomposition and personalization gap
    var_fixed = float(np.var(np.asarray(fit_s3.model.exog) @ np.asarray(fit_s3.fe_params)))
    var_total = var_fixed + var_u0_s3 + var_u1_s3 + var_e_s3
    if var_total > 0:
        r2_marginal = var_fixed / var_total
        r2_conditional = (var_fixed + var_u0_s3 + var_u1_s3) / var_total
    else:
        r2_marginal = 0.0
        r2_conditional = 0.0
    personalization_gap = r2_conditional - r2_marginal

    progressive_results[resp_name]["Stage3_random_slope"] = {
        "var_u0": var_u0_s3, "var_u1": var_u1_s3,
        "cov_u01": cov_u01_s3, "cor_u01": cor_u01_s3,
        "var_e": var_e_s3, "icc": icc_s3, "aic": fit_s3.aic,
        "lrt_stat": lrt_stat, "lrt_p_mix": lrt_p,
        "R2_marginal": r2_marginal,
        "R2_conditional": r2_conditional,
        "personalization_gap": personalization_gap,
    }
    stage3_baseline[resp_name] = {
        "fit": fit_s3, "resp_col": resp_col, "is_log": is_log,
        "df_fit": df_resp, "fe_params": dict(fit_s3.fe_params),
        "var_u0": var_u0_s3, "var_u1": var_u1_s3,
        "cov_u01": cov_u01_s3, "cor_u01": cor_u01_s3, "var_e": var_e_s3,
    }
    print(f"    var_u1 (slope) = {var_u1_s3:.6f}")
    print(f"    cor(u0, u1)    = {cor_u01_s3:.4f}")
    print(f"    LRT p (mixture) = {lrt_p:.2e}")
    print(f"    R2_m = {r2_marginal:.4f}, R2_c = {r2_conditional:.4f}, "
          f"gap = {personalization_gap:.4f}")


# blup reliability analysis
blup_dfs = {}

for resp_name, info in stage3_baseline.items():
    fit = info["fit"]
    resp_col = info["resp_col"]
    is_log = info["is_log"]
    var_u0 = info["var_u0"]
    var_u1 = info["var_u1"]
    var_e = info["var_e"]
    tag = "[LOG]" if is_log else "[RAW]"
    print(f"\n--- {tag} {resp_name} ---")

    blup_rows = []
    for pid, re_vals in fit.random_effects.items():
        u0i = float(re_vals.iloc[0])
        u1i = float(re_vals.iloc[1]) if len(re_vals) >= 2 else 0.0
        blup_rows.append({"participant_id": pid, "u0i": u0i, "u1i": u1i})
    df_blup = pd.DataFrame(blup_rows)

    meal_counts = (df_model.dropna(subset=[resp_col])
                   .groupby("participant_id").size()
                   .reset_index(name="n_meals"))
    df_blup = df_blup.merge(meal_counts, on="participant_id", how="left")

    if var_u1 > 0:
        df_blup["reliability_slope"] = var_u1 / (
            var_u1 + var_e / df_blup["n_meals"]
        )
    else:
        df_blup["reliability_slope"] = 0.0
    if var_u0 > 0:
        df_blup["reliability_intercept"] = var_u0 / (
            var_u0 + var_e / df_blup["n_meals"]
        )
    else:
        df_blup["reliability_intercept"] = 0.0

    gamma10 = fit.fe_params.get(CARB_SLOPE_VAR, 0.0)
    df_blup["total_slope"] = gamma10 + df_blup["u1i"]

    blup_dfs[resp_name] = df_blup

    n_persons = len(df_blup)
    r_slope = df_blup["reliability_slope"]
    r_int = df_blup["reliability_intercept"]
    ts = df_blup["total_slope"]

    n_pos = int((ts > 0).sum())
    n_neg = int((ts < 0).sum())

    if var_u1 > 0:
        target_r = 0.5
        n_needed = var_e * target_r / (var_u1 * (1 - target_r))
        print(f"    meals needed for slope R={target_r:.1f}: {n_needed:.1f}")

        blup_cor = df_blup[["u0i", "u1i"]].corr().iloc[0, 1]


# plot Figure 3d
ca2_response = "swt_db2_energy_cA2_log"
if ca2_response not in blup_dfs and blup_dfs:
    ca2_response = next(iter(blup_dfs))

if ca2_response in blup_dfs and ca2_response in stage3_baseline:
    df_blup = blup_dfs[ca2_response].copy()
    s3 = stage3_baseline[ca2_response]
    gamma10 = s3["fe_params"].get(CARB_SLOPE_VAR, 0.0)
    slope_sd = np.sqrt(s3["var_u1"]) if s3["var_u1"] > 0 else 0.0

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    c_main = "#E36A3A"
    c_curve = "#333333"

    total_slopes = df_blup["total_slope"].dropna().values
    n_persons = len(total_slopes)

    if n_persons > 0:
        n_bins = 50
        counts, bin_edges, patches = ax.hist(
            total_slopes, bins=n_bins, density=True,
            edgecolor="white", linewidth=0.3,
            alpha=0.9, zorder=2
        )

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "slope_orange", ["#F8D7C3", "#E87543"]
        )
        norm = mcolors.Normalize(
            vmin=float(np.min(total_slopes)),
            vmax=float(np.max(total_slopes)),
        )
        bin_width = bin_edges[1] - bin_edges[0]
        for patch, left_edge in zip(patches, bin_edges[:-1]):
            patch.set_facecolor(cmap(norm(left_edge + bin_width / 2)))

        y_theory = np.array([])
        if slope_sd > 0:
            x_theory = np.linspace(
                gamma10 - 4 * slope_sd, gamma10 + 4 * slope_sd, 200
            )
            y_theory = stats.norm.pdf(x_theory, gamma10, slope_sd)
            ax.plot(
                x_theory, y_theory,
                color=c_curve, linewidth=2.2, linestyle="--", zorder=3
            )
            ax.set_xlim(gamma10 - 4 * slope_sd, gamma10 + 4 * slope_sd)

        y_candidates = [float(np.max(counts))]
        if y_theory.size > 0:
            y_candidates.append(float(np.max(y_theory)))
        y_top = float(np.ceil(max(y_candidates) * 1.05))
        ax.set_ylim(0, y_top)

        ax.axvline(x=gamma10, color=c_main, linewidth=2.2,
                   linestyle="-", zorder=4)
        ax.text(
            gamma10, y_top * 1.035,
            f"$\\gamma_{{10}}$ = {gamma10:.3f}",
            ha="center", va="bottom", fontsize=16, fontweight="bold",
            color=c_main,
            bbox=dict(
                boxstyle="round,pad=0.18", facecolor="white",
                edgecolor=c_main, linewidth=1.1, alpha=0.95,
            ),
            clip_on=False,
        )

        ax.set_xlabel(
            "Individual Carbohydrate Sensitivity (total slope)", fontsize=18
        )
        ax.set_ylabel("Density", fontsize=18)
        ax.tick_params(axis="both", labelsize=12, width=1.0, length=4)
        ax.spines["top"].set_visible(False)
        plt.savefig("blup_distribution.png",
                    dpi=300, bbox_inches="tight")
        plt.show()
