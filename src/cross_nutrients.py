"""
Extend the random-slope framework beyond carbohydrate to fat, protein and fiber.

1. individual nutrient random slopes
  y ~ fixed_effects + (1 + nutrient_within | person), one nutrient at a time. 
 
2. pairwise nutrient random slopes
  y ~ fixed_effects + (1 + nut_A_within + nut_B_within | person). 

This script should be run after progressive_modeling.py。
"""

import numpy as np
from utils import fit_mixedlm, lrt_boundary_corrected


nutrient_cols = {
    "carb": CARB_COL,
    "protein": PROTEIN_COL,
    "fat": FAT_COL,
    "fiber": FIBER_COL,
}

PAIRWISE_PAIRS = [
    ("carb", "fat"),
    ("carb", "protein"),
    ("fat", "protein"),
]

# CWC decomposition for nutrients
available_nutrients = {}
for nut_key, nut_col in nutrient_cols.items():
    if nut_col in df_model.columns:
        available_nutrients[nut_key] = nut_col
    else:
        print(f" Nutrient column '{nut_col}' ({nut_key}) not found")
print(f"  Available nutrients: {list(available_nutrients.keys())}")


df_model[MEAL_CATEGORY_COL] = (
    df_model[MEAL_CATEGORY_COL].astype(str).str.strip().str.lower()
)

nutrient_cwc = {}
n_persons = df_model["participant_id"].nunique()

for nut_key, nut_col in available_nutrients.items():
    print(f"\n  [{nut_key.upper()}] ({nut_col})")

    grand_mean = df_model[nut_col].mean()
    grand_std = df_model[nut_col].std()

    person_means = df_model.groupby("participant_id")[nut_col].mean()
    pm_mean = person_means.mean()
    pm_std = person_means.std()

    pm_map = df_model["participant_id"].map(person_means)
    within_raw = df_model[nut_col] - pm_map

    within_var_per_person = df_model.groupby("participant_id").apply(
        lambda g: g[nut_col].var(ddof=1) if len(g) > 1 else np.nan
    )
    pooled_within_var = within_var_per_person.dropna().mean()
    pooled_within_sd = np.sqrt(pooled_within_var)

    between_var = pm_std ** 2
    within_var = pooled_within_var
    pct_between = between_var / (between_var + within_var) * 100
    pct_within = 100 - pct_between

    within_col = f"{nut_key}_within"
    between_col = f"{nut_key}_between"

    if pooled_within_sd > 1e-10:
        df_model[within_col] = within_raw / pooled_within_sd
    else:
        print("    WARNING: pooled within-SD ~ 0; using grand SD fallback.")
        df_model[within_col] = within_raw / grand_std if grand_std > 0 else 0.0

    if pm_std > 1e-10:
        df_model[between_col] = (pm_map - pm_mean) / pm_std
    else:
        df_model[between_col] = 0.0

    per_person_wsd = df_model.groupby("participant_id")[nut_col].std()
    n_low_var = (per_person_wsd < 5).sum()
    if n_low_var > 0:
        print(f"    Persons with within-SD < 5g: {n_low_var}/{n_persons} "
              f"({n_low_var / n_persons * 100:.1f}%)")

    nutrient_cwc[nut_key] = {
        "raw_col": nut_col,
        "within_col": within_col,
        "between_col": between_col,
        "pooled_w_sd": pooled_within_sd,
        "pct_between": pct_between,
        "pct_within": pct_within,
    }

# formula contruction
fe_parts = []
for nut_key in available_nutrients:
    fe_parts.append(nutrient_cwc[nut_key]["within_col"])
    fe_parts.append(nutrient_cwc[nut_key]["between_col"])
fe_parts.append("has_alcohol")

FORMULA_FE_BASE = " + ".join(fe_parts)
FORMULA_FE = FORMULA_FE_BASE + f" + C({MEAL_CATEGORY_COL})"
print(f"  FE formula: {FORMULA_FE}")

COLS_NEEDED = list(set(
    [nutrient_cwc[nk]["within_col"] for nk in available_nutrients]
    + [nutrient_cwc[nk]["between_col"] for nk in available_nutrients]
    + ["participant_id", "has_alcohol", MEAL_CATEGORY_COL]
))


# reference model with all available nutrients as fixed effects and random intercept only
ri_results = {}
for resp_col, resp_disp, is_log in response_runs:
    df_fit = df_model[COLS_NEEDED + [resp_col]].dropna().copy()
    fit, status = fit_mixedlm(
        f"{resp_col} ~ {FORMULA_FE}",
        df_fit,
        groups="participant_id",
        re_formula="~1",
    )

    var_u0 = float(fit.cov_re.iloc[0, 0])
    var_e = float(fit.scale)
    icc = var_u0 / (var_u0 + var_e)
    ri_results[resp_col] = {
        "fit": fit,
        "df_fit": df_fit,
        "var_u0": var_u0,
        "var_e": var_e,
        "icc": icc,
        "llf": fit.llf,
        "n_obs": len(df_fit),
        "n_grp": df_fit["participant_id"].nunique(),
        "status": status,
    }
    print(f"  {resp_disp}: n={len(df_fit)}, "
          f"grp={ri_results[resp_col]['n_grp']}, ICC={icc:.4f}, "
          f"var_u0={var_u0:.6f}, var_e={var_e:.6f} [{status}]")


# individual nutrient random slope models
individual_nut_results = {}

for nut_key in available_nutrients:
    slope_var = nutrient_cwc[nut_key]["within_col"]
    print(f"\n  NUTRIENT: {nut_key.upper()} (random slope on '{slope_var}')")
    individual_nut_results[nut_key] = {}

    for resp_col, resp_disp, is_log in response_runs:
        if resp_col not in ri_results:
            continue

        ref = ri_results[resp_col]
        df_fit = ref["df_fit"].copy()
        if slope_var not in df_fit.columns or df_fit[slope_var].std() < 1e-10:
            print(f"    {resp_disp}: slope var zero variance - skipped")
            individual_nut_results[nut_key][resp_col] = {"status": "SKIPPED_NO_VARIANCE"}
            continue

        fit, status = fit_mixedlm(
            f"{resp_col} ~ {FORMULA_FE}",
            df_fit,
            groups="participant_id",
            re_formula=f"~{slope_var}",
        )

        re_cov = fit.cov_re
        labels = list(re_cov.index)
        var_u0 = float(re_cov.iloc[0, 0])
        var_e = float(fit.scale)

        if slope_var in labels:
            idx = labels.index(slope_var)
            var_u1 = float(re_cov.iloc[idx, idx])
            cov_u01 = float(re_cov.iloc[0, idx])
            sd0 = np.sqrt(max(var_u0, 0))
            sd1 = np.sqrt(max(var_u1, 0))
            cor_u01 = cov_u01 / (sd0 * sd1) if sd0 > 0 and sd1 > 0 else np.nan
        else:
            var_u1 = 0.0
            cov_u01 = 0.0
            cor_u01 = np.nan
            sd1 = 0.0

        gamma = fit.fe_params.get(slope_var, np.nan)
        lrt_stat, lrt_p_mix = lrt_boundary_corrected(fit.llf, ref["llf"], n_extra=2)
        sig = "***" if lrt_p_mix < 0.001 else "**" if lrt_p_mix < 0.01 else "*" if lrt_p_mix < 0.05 else "ns"

        individual_nut_results[nut_key][resp_col] = {
            "status": status,
            "fit": fit,
            "var_u0": var_u0,
            "var_u1": var_u1,
            "cov_u01": cov_u01,
            "cor_u01": cor_u01,
            "var_e": var_e,
            "gamma": gamma,
            "slope_sd": sd1,
            "range_lo": gamma - 1.96 * sd1,
            "range_hi": gamma + 1.96 * sd1,
            "lrt_stat": lrt_stat,
            "lrt_p_mix": lrt_p_mix,
            "llf": fit.llf,
            "n_obs": ref["n_obs"],
            "n_grp": ref["n_grp"],
        }


# pairwise nutrient random slope models and cross-nutrient slope correlation analysis
cross_nut_results = {}

for nut_a, nut_b in PAIRWISE_PAIRS:
    if nut_a not in available_nutrients or nut_b not in available_nutrients:
        print(f"\n  [{nut_a.upper()} + {nut_b.upper()}]: unavailable - skipped")
        continue

    sv_a = nutrient_cwc[nut_a]["within_col"]
    sv_b = nutrient_cwc[nut_b]["within_col"]
    pair_key = f"{nut_a}_{nut_b}"

    print(f"\n  PAIR: {nut_a.upper()} + {nut_b.upper()}")
    print(f"  re_formula = ~{sv_a} + {sv_b}")

    cross_nut_results[pair_key] = {}

    for resp_col, resp_disp, is_log in response_runs:
        if resp_col not in ri_results:
            continue

        ref = ri_results[resp_col]
        df_fit = ref["df_fit"].copy()

        fit, status = fit_mixedlm(
            f"{resp_col} ~ {FORMULA_FE}",
            df_fit,
            groups="participant_id",
            re_formula=f"~{sv_a} + {sv_b}",
        )

        re_cov = fit.cov_re
        labels = list(re_cov.index)
        var_u0 = float(re_cov.iloc[0, 0])
        var_e = float(fit.scale)

        var_u1_a = float(re_cov.iloc[labels.index(sv_a), labels.index(sv_a)]) if sv_a in labels else 0.0
        var_u1_b = float(re_cov.iloc[labels.index(sv_b), labels.index(sv_b)]) if sv_b in labels else 0.0

        if sv_a in labels and sv_b in labels:
            idx_a = labels.index(sv_a)
            idx_b = labels.index(sv_b)
            cov_ab = float(re_cov.iloc[idx_a, idx_b])
            sd_a = np.sqrt(max(var_u1_a, 0))
            sd_b = np.sqrt(max(var_u1_b, 0))
            cor_ab = cov_ab / (sd_a * sd_b) if sd_a > 0 and sd_b > 0 else np.nan
        else:
            cov_ab = np.nan
            cor_ab = np.nan

        lrt_stat, lrt_p_mix = lrt_boundary_corrected(fit.llf, ref["llf"], n_extra=5)

        lrt_vs_a = {}
        r_single_a = individual_nut_results.get(nut_a, {}).get(resp_col, {})
        if "llf" in r_single_a:
            stat_a, p_a = lrt_boundary_corrected(fit.llf, r_single_a["llf"], n_extra=2)
            lrt_vs_a = {"stat": stat_a, "p_mix": p_a}

        lrt_vs_b = {}
        r_single_b = individual_nut_results.get(nut_b, {}).get(resp_col, {})
        if "llf" in r_single_b:
            stat_b, p_b = lrt_boundary_corrected(fit.llf, r_single_b["llf"], n_extra=2)
            lrt_vs_b = {"stat": stat_b, "p_mix": p_b}

        sig = "***" if lrt_p_mix < 0.001 else "**" if lrt_p_mix < 0.01 else "*" if lrt_p_mix < 0.05 else "ns"

        cross_nut_results[pair_key][resp_col] = {
            "status": status,
            "fit": fit,
            "var_u0": var_u0,
            "var_u1_a": var_u1_a,
            "var_u1_b": var_u1_b,
            "cor_slopes": cor_ab,
            "cov_slopes": cov_ab,
            "var_e": var_e,
            "lrt_stat": lrt_stat,
            "lrt_p_mix": lrt_p_mix,
            "lrt_vs_a": lrt_vs_a,
            "lrt_vs_b": lrt_vs_b,
        }
