"""
Cross-level moderator models for carbohydrate sensitivity.

This script should be run after progressive_modeling.py.
"""

import gc

import numpy as np
import pandas as pd
import patsy
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from utils import (
    extract_variance_components,
    fit_mixedlm,
)

RESPONSE_GBLUP_KEYS = {
    "swt_db2_energy_cA2_log": "gblup_cA2",
    "swt_db2_energy_cD2_log": "gblup_cD2",
    "swt_db2_energy_cD1_log": "gblup_cD1",
    "iauc_log": "gblup_iAUC",
    "peak_rise_log": "gblup_peak_rise",
    "peak_time_raw": "gblup_peak_time",
}


# individual variable test
individual_tests = []

demo_main, demo_interact = [], []
demo_main.append(zscore_vars["age"])
demo_interact.append(zscore_vars["age"])
demo_main.append("sex_binary")
demo_interact.append("sex_binary")
individual_tests.append(("Demographics", demo_main, demo_interact, "age + sex"))

single_tests = [
    ("BMI", "bmi", "BMI"),
    ("WaistHipRatio", "waist_to_hip_ratio", "waist-to-hip ratio"),
    (
        "BodyCompFat",
        "body_comp_total_tissue_percent_fat",
        "total tissue percent fat",
    ),
    ("VAT", "total_scan_vat_mass", "visceral fat mass"),
    ("HbA1c", "bt__hba1c_float_value", "HbA1c"),
]
for test_name, var_key, desc in single_tests:
    if var_key in zscore_vars:
        z_col = zscore_vars[var_key]
        individual_tests.append((test_name, [z_col], [z_col], desc))

micro_main = [zscore_vars[key] for key in ["pco1", "pco2"] if key in zscore_vars]
individual_tests.append(("Microbiome", micro_main, micro_main, "gut microbiome PCoA")    )

individual_tests_by_response = {}
for resp_name in stage3_baseline:
    tests_for_resp = list(individual_tests)
    gblup_key = RESPONSE_GBLUP_KEYS.get(resp_name)
    if gblup_key and gblup_key in zscore_vars:
        z_col = zscore_vars[gblup_key]
        tests_for_resp.append(
            ("gBLUP", [z_col], [z_col], f"gBLUP({gblup_key})")
        )
    individual_tests_by_response[resp_name] = tests_for_resp


# phase A independent screening
phase_a_results = {}
phase_a_diagnostics = []

for resp_name, baseline_info in stage3_baseline.items():
    resp_col = baseline_info["resp_col"]
    is_log = baseline_info["is_log"]

    phase_a_results[resp_name] = {}

    baseline_tau2_slope = baseline_info["var_u1"]
    baseline_tau2_intercept = baseline_info["var_u0"]

    for test_name, main_vars, interact_vars, _ in individual_tests_by_response[resp_name]:
        gc.collect()

        main_terms = " + ".join(main_vars)
        interact_terms = " + ".join(
            [f"{CARB_SLOPE_VAR}:{v}" for v in interact_vars]
        )
        formula = (
            f"{resp_col} ~ {FORMULA_NUTRIENTS} + "
            f"{main_terms} + {interact_terms}"
        )

        cols_needed = list(dict.fromkeys([
            resp_col,
            CARB_SLOPE_VAR,
            "participant_id",
            PROTEIN_COL,
            FAT_COL,
            FIBER_COL,
            "has_alcohol",
            "carb_between",
            MEAL_CATEGORY_COL] + main_vars))
        df_fit = df_model.dropna(subset=cols_needed).copy()
        mc = df_fit.groupby("participant_id").size()
        valid = mc[mc >= MIN_MEALS].index
        df_fit = df_fit[df_fit["participant_id"].isin(valid)].copy()

        n_obs = len(df_fit)
        n_groups = df_fit["participant_id"].nunique()

        fit, status = fit_mixedlm(
            formula,
            data=df_fit,
            groups=df_fit["participant_id"],
            re_formula=f"~{CARB_SLOPE_VAR}",
        )

        if fit is None:
            print("  CONVERGENCE FAILED. SKIPPING.")
            phase_a_results[resp_name][test_name] = {
                "status": "FAILED",
                "error": status,
                "n_obs": n_obs,
                "n_groups": n_groups,
            }
            continue

        if not str(status).startswith("converged"):
            print(f"  Convergence note: {status}")

        var_u0, var_u1, cov_u01, cor_u01, var_e = extract_variance_components(
            fit,
            has_random_slope=True,
        )

        pct_tau2_slope = (
            100 * (var_u1 - baseline_tau2_slope) / baseline_tau2_slope
            if baseline_tau2_slope > 0 else 0.0
        )
        pct_tau2_int = (
            100 * (var_u0 - baseline_tau2_intercept) / baseline_tau2_intercept
            if baseline_tau2_intercept > 0 else 0.0
        )

        interaction_results = {}
        for v in interact_vars:
            interact_term = f"{CARB_SLOPE_VAR}:{v}"
            if interact_term not in fit.fe_params.index:
                continue

            coef = fit.fe_params[interact_term]
            se = fit.bse[interact_term]
            z_stat = fit.tvalues[interact_term]
            p_val = fit.pvalues[interact_term]

            interaction_results[interact_term] = {
                "coef": float(coef),
                "se": float(se),
                "z": float(z_stat),
                "p": float(p_val),
            }

        result = {
            "status": status,
            "var_u0": var_u0,
            "var_u1": var_u1,
            "cov_u01": cov_u01,
            "cor_u01": cor_u01,
            "var_e": var_e,
            "llf": fit.llf,
            "n_obs": n_obs,
            "n_groups": n_groups,
            "n_persons": n_groups,
            "pct_tau2_slope_from_baseline": pct_tau2_slope,
            "pct_tau2_intercept_from_baseline": pct_tau2_int,
            "tau2_slope_pct_of_baseline": (
                100 * var_u1 / baseline_tau2_slope
                if baseline_tau2_slope > 0 else np.nan
            ),
            "formula": formula,
            "fe_params": fit.fe_params.to_dict(),
            "fe_pvalues": fit.pvalues.to_dict(),
            "interaction_results": interaction_results,
            "main_vars": list(main_vars),
            "interact_vars": list(interact_vars),
        }
        phase_a_results[resp_name][test_name] = result

        phase_a_diagnostics.append({
            "response": resp_name,
            "is_log": is_log,
            "test": test_name,
            "n_obs": n_obs,
            "n_groups": n_groups,
            "tau2_slope": var_u1,
            "tau2_slope_pct_of_baseline": result["tau2_slope_pct_of_baseline"],
            "tau2_intercept": var_u0,
            "var_e": var_e,
            "cor_u01": cor_u01,
        })

df_phase_a = pd.DataFrame(phase_a_diagnostics)

# phase b joint models 
PHASE_B_SIG_THRESHOLD = 0.05

phase_b_candidates = {}

for resp_name in stage3_baseline.keys():
    if resp_name not in phase_a_results:
        phase_b_candidates[resp_name] = []
        continue

    candidates = []
    for test_name, test_result in phase_a_results[resp_name].items():
        if not str(test_result.get("status", "")).startswith("converged"):
            continue

        for term, info in test_result.get("interaction_results", {}).items():
            if ":" not in term:
                continue

            moderator_var = term.split(":", 1)[1]

            p_val = info.get("p", 1.0)
            if pd.notna(p_val) and p_val < PHASE_B_SIG_THRESHOLD:
                candidates.append({
                    "moderator_var": moderator_var,
                    "source_test": test_name,
                    "phase_a_p": float(p_val),
                    "phase_a_coef": float(info.get("coef", np.nan)),
                    "phase_a_se": float(info.get("se", np.nan)),
                })

    seen = set()
    deduped = []
    for c in candidates:
        if c["moderator_var"] not in seen:
            seen.add(c["moderator_var"])
            deduped.append(c)
    phase_b_candidates[resp_name] = deduped


phase_b_results = {}
phase_b_all_pvals = []

for resp_name, candidates in phase_b_candidates.items():
    baseline_info = stage3_baseline[resp_name]
    resp_col = baseline_info["resp_col"]
    tag = "[LOG]" if baseline_info["is_log"] else "[RAW]"

    moderators = [c["moderator_var"] for c in candidates]
    print(f"\n--- Phase B joint model {tag} {resp_name} ---")
    print(f"  Candidate interaction moderators: {moderators if moderators else 'none'}")

    if not candidates:
        phase_b_results[resp_name] = {
            "status": "SKIPPED",
            "reason": "no_phase_b_candidates",
            "moderators": [],
        }
        print("  No Phase B candidates. Skipping joint model.")
        continue

    interact_terms = [f"{CARB_SLOPE_VAR}:{m}" for m in moderators]
    all_main = []
    for v in moderators:
        if v not in all_main:
            all_main.append(v)

    main_terms_str = " + ".join(all_main)
    interact_terms_str = " + ".join(interact_terms)

    formula = (
        f"{resp_col} ~ {FORMULA_NUTRIENTS} "
        f"+ {main_terms_str} + {interact_terms_str}"
    )

    cols_needed = list(dict.fromkeys([
        "participant_id",
        CARB_SLOPE_VAR,
        resp_col,
        PROTEIN_COL,
        FAT_COL,
        FIBER_COL,
        "has_alcohol",
        "carb_between",
        MEAL_CATEGORY_COL] + all_main))
    df_fit = df_model.dropna(subset=cols_needed).copy()
    mc = df_fit.groupby("participant_id").size()
    valid_persons = mc[mc >= MIN_MEALS].index
    df_fit = df_fit[df_fit["participant_id"].isin(valid_persons)].copy()

    n_obs = len(df_fit)
    n_groups = df_fit["participant_id"].nunique()
 
    fit, status = fit_mixedlm(
        formula=formula,
        data=df_fit,
        groups=df_fit["participant_id"],
        re_formula=f"~{CARB_SLOPE_VAR}",
    )

    if fit is None or not str(status).startswith("converged"):
        print(f"  CONVERGENCE FAILED: {status}")
        phase_b_results[resp_name] = {
            "status": "FAILED",
            "error": status,
            "formula": formula,
            "moderators": moderators,
            "n_obs": n_obs,
            "n_groups": n_groups,
        }
        continue

    var_u0, var_u1, cov_u01, cor_u01, var_e = extract_variance_components(
        fit,
        has_random_slope=True,
    )
    baseline_tau2 = baseline_info.get("var_u1", np.nan)
    pct_tau2 = (
        100 * var_u1 / baseline_tau2
        if pd.notna(baseline_tau2) and baseline_tau2 > 0 else np.nan
    )
    print(f"  tau2_slope: {var_u1:.6f} ({pct_tau2:.1f}% of baseline)")

    interaction_results = {}
    print("  Interaction coefficients (mutually adjusted):")
    for c in candidates:
        moderator_var = c["moderator_var"]
        term = f"{CARB_SLOPE_VAR}:{moderator_var}"
        if term not in fit.fe_params.index:
            interaction_results[term] = {
                "status": "MISSING",
                "moderator_var": moderator_var,
                "source_test": c["source_test"],
            }
            continue

        coef = float(fit.fe_params[term])
        se = float(fit.bse[term])
        p_val = float(fit.pvalues[term])
        phase_a_coef = c["phase_a_coef"]
        if pd.notna(phase_a_coef) and phase_a_coef != 0:
            atten_pct = (phase_a_coef - coef) / phase_a_coef * 100.0
        else:
            atten_pct = np.nan
        sig = ("***" if p_val < 0.001 else
               "**" if p_val < 0.01 else
               "*" if p_val < 0.05 else "ns")

        print(
            f"    {term:<45s} beta={coef:+.5f} SE={se:.5f} "
            f"p={p_val:.4g} ({sig}); "
            f"Phase A beta={phase_a_coef:+.5f}, p={c['phase_a_p']:.4g}, "
            f"atten={atten_pct:+.1f}%"
        )

        interaction_results[term] = {
            "moderator_var": moderator_var,
            "source_test": c["source_test"],
            "coef": coef,
            "se": se,
            "p": p_val,
            "phase_a_coef": phase_a_coef,
            "phase_a_p": c["phase_a_p"],
            "phase_a_se": c["phase_a_se"],
            "attenuation_pct": atten_pct,
        }
        phase_b_all_pvals.append({
            "response": resp_name,
            "moderator_var": moderator_var,
            "source_test": c["source_test"],
            "p": p_val,
            "coef": coef,
            "se": se,
            "phase_a_p": c["phase_a_p"],
            "phase_a_coef": phase_a_coef,
            "attenuation_pct": atten_pct,
        })

    null_formula = f"{resp_col} ~ {FORMULA_NUTRIENTS} + {main_terms_str}"
    fit_null, status_null = fit_mixedlm(
        formula=null_formula,
        data=df_fit,
        groups=df_fit["participant_id"],
        re_formula=f"~{CARB_SLOPE_VAR}",
    )
    if fit_null is not None and str(status_null).startswith("converged"):
        ll_full = fit.llf
        ll_null = fit_null.llf
        lrt_stat = max(2.0 * (ll_full - ll_null), 0.0)
        df_lrt = len(moderators)
        lrt_p = float(chi2.sf(lrt_stat, df_lrt)) if lrt_stat > 0 else 1.0
    else:
        lrt_stat = np.nan
        df_lrt = len(moderators)
        lrt_p = np.nan

    print(
        f"  Omnibus LRT for all Phase B interactions: "
        f"chi2({df_lrt})={lrt_stat:.3f}, p={lrt_p:.4g}"
    )

    phase_b_results[resp_name] = {
        "status": "OK",
        "n_obs": n_obs,
        "n_groups": n_groups,
        "n_persons": n_groups,
        "moderators": moderators,
        "all_main_terms": all_main,
        "var_u0": var_u0,
        "var_u1": var_u1,
        "cov_u01": cov_u01,
        "cor_u01": cor_u01,
        "var_e": var_e,
        "pct_tau2_slope": pct_tau2,
        "tau2_slope_pct_of_baseline": pct_tau2,
        "llf": fit.llf,
        "interaction_results": interaction_results,
        "joint_interaction_results": interaction_results,
        "lrt_stat": lrt_stat,
        "lrt_df": df_lrt,
        "lrt_p": lrt_p,
        "formula": formula,
        "fe_params": fit.fe_params.to_dict(),
        "fe_pvalues": fit.pvalues.to_dict(),
    }
    del fit
    if fit_null is not None:
        del fit_null

if not phase_b_all_pvals:
    print("  No Phase B interactions tested. Nothing to correct.")
    df_phase_b = pd.DataFrame()
else:
    pvals_arr = np.array([x["p"] for x in phase_b_all_pvals], dtype=float)
    pvals_arr = np.where(np.isfinite(pvals_arr), pvals_arr, 1.0)
    _, pvals_fdr, _, _ = multipletests(
        pvals_arr,
        method="fdr_bh",
        alpha=0.05,
    )

    for i, entry in enumerate(phase_b_all_pvals):
        entry["p_fdr"] = float(pvals_fdr[i])
        entry["fdr_sig"] = bool(pvals_fdr[i] < 0.05)

    n_total = len(phase_b_all_pvals)
    n_uncorrected = sum(1 for x in phase_b_all_pvals if x["p"] < 0.05)
    n_fdr = sum(1 for x in phase_b_all_pvals if x["fdr_sig"])

    print(f"\n  Total interactions tested: {n_total}")
    print(f"  Significant at p<0.05 (uncorrected): {n_uncorrected}/{n_total}")
    print(f"  Significant at FDR<0.05:             {n_fdr}/{n_total}")

    sorted_entries = sorted(phase_b_all_pvals, key=lambda x: x["p"])
    print(
        f"\n  {'Response':<30s} {'Moderator':<32s} "
        f"{'Phase A p':>12s} {'Phase B p':>12s} {'p_FDR':>12s} "
        f"{'Atten%':>8s} Sig"
    )

    for e in sorted_entries:
        if e["fdr_sig"]:
            sig = "** FDR"
        elif e["p"] < 0.05:
            sig = "* nom"
        else:
            sig = "ns"
        atten = e.get("attenuation_pct", np.nan)
        atten_str = f"{atten:+.1f}" if not pd.isna(atten) else "NA"
        print(
            f"  {e['response']:<30s} {e['moderator_var']:<32s} "
            f"{e['phase_a_p']:>12.4g} {e['p']:>12.4g} "
            f"{e['p_fdr']:>12.4g} {atten_str:>8s} {sig}"
        )

    df_phase_b = pd.DataFrame(phase_b_all_pvals)


# plot figure 4a
resp_order = [
    "swt_db2_energy_cA2_log",
    "swt_db2_energy_cD2_log",
    "swt_db2_energy_cD1_log",
    "peak_rise_log",
    "iauc_log",
    "peak_time_raw",
]
resp_labels = [
    "cA2 Energy (low-freq)",
    "cD2 Energy (mid-freq)",
    "cD1 Energy (high-freq)",
    "Peak Rise",
    "iAUC",
    "Peak Time",
]

mod_order = [
    "age_z",
    "sex_binary",
    "bmi_z",
    "waist_to_hip_ratio_z",
    "body_comp_total_tissue_percent_fat_z",
    "total_scan_vat_mass_z",
    "bt__hba1c_float_value_z",
    "pco1_z",
    "pco2_z",
    "gBLUP_z",
]
mod_labels = [
    "Age",
    "Sex",
    "BMI",
    "Waist-Hip Ratio",
    "Body Fat %",
    "VAT Mass",
    "HbA1c",
    "PCo1",
    "PCo2",
    "gBLUP",
]

coef_matrix = np.full((len(mod_order), len(resp_order)), np.nan)
pval_matrix = np.full((len(mod_order), len(resp_order)), np.nan)

for resp_name in phase_a_results:
    if resp_name not in resp_order:
        continue
    j = resp_order.index(resp_name)

    for mi, mod_col in enumerate(mod_order):
        if mod_col == "gBLUP_z":
            gblup_entry = phase_a_results[resp_name].get("gBLUP", {})
            if str(gblup_entry.get("status", "")).startswith("converged"):
                for term, info in gblup_entry.get("interaction_results", {}).items():
                    coef_matrix[mi, j] = info["coef"]
                    pval_matrix[mi, j] = info["p"]
                    break
        else:
            target_term = f"{CARB_SLOPE_VAR}:{mod_col}"
            for test_name, res in phase_a_results[resp_name].items():
                if test_name == "gBLUP":
                    continue
                if not str(res.get("status", "")).startswith("converged"):
                    continue
                interactions = res.get("interaction_results", {})
                if target_term in interactions:
                    info = interactions[target_term]
                    coef_matrix[mi, j] = info["coef"]
                    pval_matrix[mi, j] = info["p"]
                    break

coef_T = coef_matrix.T
pval_T = pval_matrix.T

signed_sig = np.zeros_like(coef_T)
for i in range(coef_T.shape[0]):
    for j in range(coef_T.shape[1]):
        if np.isnan(pval_T[i, j]):
            signed_sig[i, j] = 0
        else:
            logp = -np.log10(max(pval_T[i, j], 1e-10))
            signed_sig[i, j] = np.sign(coef_T[i, j]) * logp

signed_sig = np.clip(signed_sig, -4, 4)

fig, ax = plt.subplots(figsize=(13, 5))

cmap = plt.cm.RdBu_r
norm = mcolors.TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
im = ax.imshow(signed_sig, cmap=cmap, norm=norm, aspect="auto")

for i in range(len(resp_order)):
    for j in range(len(mod_order)):
        if np.isnan(pval_T[i, j]):
            ax.text(
                j, i, "-",
                ha="center", va="center", fontsize=8, color="#DDDDDD"
            )
        else:
            p = pval_T[i, j]
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            else:
                stars = ""
            if stars:
                txt_color = "white" if abs(signed_sig[i, j]) > 2.5 else "#333333"
                ax.text(
                    j, i, stars,
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color=txt_color,
                )

ax.set_xticks(np.arange(len(mod_order)))
ax.set_xticklabels(mod_labels, fontsize=10, ha="center")
ax.xaxis.set_ticks_position("bottom")

ax.set_yticks(np.arange(len(resp_order)))
ax.set_yticklabels(resp_labels, fontsize=10.5)

ax.axhline(y=2.5, color="white", linewidth=2.5)

for xpos in [1.5, 3.5, 5.5, 6.5, 8.5]:
    ax.axvline(x=xpos, color="white", linewidth=1.5)

cat_labels = {
    "Demographics": 0.5,
    "Anthropometrics": 2.5,
    "Body Composition": 4.5,
    "Glycemic": 6.0,
    "Microbiome": 7.5,
    "Genetic": 9.0,
}
for cat, xpos in cat_labels.items():
    ax.text(xpos, -0.8, cat, ha="center", va="bottom", fontsize=10)

cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.12, orientation="vertical")
cbar.set_label("sign(coef) x $-\\log_{10}(p)$", fontsize=9)
cbar.set_ticks([-4, -2, 0, 2, 4])
cbar.set_ticklabels(["-4\n(negative)", "-2", "0\n(n.s.)", "2", "4\n(positive)"])
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(
    "figure3_interaction_micro_gblup_heatmap.png",
    dpi=300, bbox_inches="tight",
)
plt.show()


# plot figure 4b
resp_labels_short = ["cA2", "cD2", "cD1", "Peak Rise", "iAUC", "Peak Time"]

b_moderators = [
    "bmi_z",
    "waist_to_hip_ratio_z",
    "body_comp_total_tissue_percent_fat_z",
    "total_scan_vat_mass_z",
    "pco1_z",
]
mod_display = {
    "bmi_z": "BMI",
    "waist_to_hip_ratio_z": "Waist-Hip Ratio",
    "body_comp_total_tissue_percent_fat_z": "Body Fat %",
    "total_scan_vat_mass_z": "VAT Mass",
    "pco1_z": "Microbiome PCo1",
}
mod_colors = {
    "bmi_z": "#3B6E8F",
    "waist_to_hip_ratio_z": "#6B9DBF",
    "body_comp_total_tissue_percent_fat_z": "#E8A063",
    "total_scan_vat_mass_z": "#B53289",
    "pco1_z": "#5BA572",
}

resp_sd = {}
for resp_name in resp_order:
    if resp_name in stage3_baseline:
        resp_col = stage3_baseline[resp_name]["resp_col"]
        resp_sd[resp_name] = df_model[resp_col].std()

rows = []
for entry in phase_b_all_pvals:
    resp_name = entry["response"]
    mod_col = entry["moderator_var"]
    if resp_name not in resp_order or mod_col not in mod_display:
        continue
    sd_y = resp_sd.get(resp_name, np.nan)
    if pd.isna(sd_y) or sd_y == 0:
        continue

    rows.append({
        "resp": resp_labels_short[resp_order.index(resp_name)],
        "resp_name": resp_name,
        "mod": mod_col,
        "mod_label": mod_display[mod_col],
        "coef_a": entry["phase_a_coef"] / sd_y,
        "p_a": entry["phase_a_p"],
        "coef_b": entry["coef"] / sd_y,
        "p_b": entry["p"],
        "p_fdr": entry.get("p_fdr", np.nan),
        "sig_a": entry["phase_a_p"] < 0.05,
        "sig_b": entry["p"] < 0.05,
        "fdr_sig": entry.get("fdr_sig", False),
    })

rows_sig = [r for r in rows if r["sig_a"]]
rows_sig.sort(
    key=lambda r: (
        b_moderators.index(r["mod"]),
        resp_order.index(r["resp_name"]),
    )
)
print(f"Plotting {len(rows_sig)} Phase A significant interactions")

if rows_sig:
    fig, ax = plt.subplots(figsize=(10, 7))

    y_gap = 0
    y_coords = []
    current_mod = None
    divider_positions = []
    for i, row in enumerate(rows_sig):
        if current_mod is not None and row["mod"] != current_mod:
            y_gap += 0.8
            divider_positions.append(y_gap + i - 0.4)
        current_mod = row["mod"]
        y_coords.append(i + y_gap)
    y_coords = np.array(y_coords)

    all_coefs = [r["coef_a"] for r in rows_sig]
    all_coefs += [r["coef_b"] for r in rows_sig if not np.isnan(r["coef_b"])]
    x_span = max(all_coefs) - min(all_coefs)
    x_margin = x_span * 0.15 if x_span > 0 else 0.05
    x_min = min(min(all_coefs) - x_margin, -0.01)
    x_max = max(all_coefs) + x_margin

    for i, row in enumerate(rows_sig):
        y = y_coords[i]
        color = mod_colors[row["mod"]]

        ax.scatter(
            row["coef_a"], y,
            color=color, marker="o", s=80,
            edgecolors="white", linewidth=0.8, zorder=3,
        )

        if not np.isnan(row["coef_b"]):
            alpha_b = 1.0 if row["sig_b"] else 0.4
            ax.scatter(
                row["coef_b"], y,
                color=color, marker="D", s=70,
                edgecolors="white", linewidth=0.8,
                alpha=alpha_b, zorder=3,
            )
            line_alpha = 0.8 if row["sig_b"] else 0.3
            ax.plot(
                [row["coef_a"], row["coef_b"]], [y, y],
                color=color, linewidth=1.5, alpha=line_alpha, zorder=2,
            )

        ax.text(
            x_min - (x_max - x_min) * 0.02, y, row["resp"],
            ha="right", va="center", fontsize=9, color="#555555",
        )

        if row["sig_b"]:
            rightmost = max(row["coef_a"], row["coef_b"])
            ax.text(
                rightmost + (x_max - x_min) * 0.02, y,
                "survives", ha="left", va="center", fontsize=7,
                color=color, fontstyle="italic", fontweight="bold",
            )
        elif not np.isnan(row["coef_b"]):
            rightmost = max(row["coef_a"], row["coef_b"])
            ax.text(
                rightmost + (x_max - x_min) * 0.02, y,
                "absorbed", ha="left", va="center", fontsize=7,
                color="#AAAAAA", fontstyle="italic",
            )

    current_mod = None
    group_ys = []
    for i, row in enumerate(rows_sig):
        if row["mod"] != current_mod:
            if current_mod is not None:
                mid_y = np.mean(group_ys)
                ax.text(
                    x_min - (x_max - x_min) * 0.12, mid_y,
                    mod_display[current_mod],
                    ha="right", va="center", fontsize=10.5,
                    fontweight="bold", color=mod_colors[current_mod],
                )
            current_mod = row["mod"]
            group_ys = [y_coords[i]]
        else:
            group_ys.append(y_coords[i])

    mid_y = np.mean(group_ys)
    ax.text(
        x_min - (x_max - x_min) * 0.12, mid_y,
        mod_display[current_mod],
        ha="right", va="center", fontsize=10.5,
        fontweight="bold", color=mod_colors[current_mod],
    )

    for d in divider_positions:
        ax.axhline(y=d, color="#EEEEEE", linewidth=0.8)

    ax.axvline(x=0, color="#CCCCCC", linewidth=0.8, linestyle="-")

    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#888888",
            markersize=9, label="Phase A (independent)",
        ),
        Line2D(
            [0], [0], marker="D", color="w", markerfacecolor="#888888",
            markersize=8, label="Phase B (jointly adjusted)",
        ),
        Line2D(
            [0], [0], marker="D", color="w", markerfacecolor="#888888",
            markersize=8, alpha=0.3, label="Phase B (n.s.)",
        ),
    ]
    ax.legend(
        handles=legend_elements, fontsize=8.5, loc="lower right",
        framealpha=0.95, edgecolor="#CCCCCC",
    )

    ax.set_xlabel(
        "Standardized Interaction Coefficient\n"
        "(effect per 1-SD moderator change, in SD units of response)",
        fontsize=10,
    )
    ax.set_xlim(x_min, x_max * 1.15)
    ax.set_ylim(max(y_coords) + 0.8, -0.8)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        "figure4_coefficient_attenuation_micro_gblup.png",
        dpi=300, bbox_inches="tight",
    )
    plt.show()


# plot figure 4c
wavelet_responses = [
    "swt_db2_energy_cA2_log",
    "swt_db2_energy_cD2_log",
    "swt_db2_energy_cD1_log",
]
wavelet_panel_titles = ["cA2 Energy", "cD2 Energy", "cD1 Energy"]

r2_moderator_order = [
    "Demographics",
    "BMI",
    "WaistHipRatio",
    "BodyCompFat",
    "VAT",
    "HbA1c",
    "Microbiome",
    "gBLUP",
]
r2_moderator_display = [
    "Demographics",
    "BMI",
    "Waist-Hip",
    "Body Fat %",
    "VAT Mass",
    "HbA1c",
    "Microbiome\nPCo1",
    "gBLUP",
]

c_fixed_dark = "#6AAB5E"
c_gap_dark = "#3B6E8F"
c_fixed_light = "#8FCC84"
c_gap_light = "#6A9DBF"
figure4c_output = "figure_r2_all_moderators_wavelet_gblup.png"

r2_rows = []
r2_skipped = []

for resp_name in wavelet_responses:
    if resp_name not in stage3_baseline:
        r2_skipped.append((resp_name, "baseline_stage3", "missing stage3 baseline"))
        continue

    s3 = stage3_baseline[resp_name]
    resp_col = s3.get("resp_col", resp_name)

    try:
        formula_str = f"{resp_col} ~ {FORMULA_NUTRIENTS}"
        fe_dict = s3["fe_params"]
        _, x_mat = patsy.dmatrices(
            formula_str,
            df_model,
            return_type="dataframe",
            NA_action="drop",
        )
        if x_mat.shape[0] == 0:
            raise ValueError("Patsy returned 0 rows after NA drop")
        missing = set(x_mat.columns) - set(fe_dict.keys())
        if missing:
            raise ValueError(f"Missing fe_params for: {sorted(missing)}")
        beta = np.array([fe_dict.get(col, 0.0) for col in x_mat.columns], dtype=float)
        var_fixed = float(np.var(x_mat.values @ beta, ddof=0))
        var_random = float(s3["var_u0"]) + float(s3["var_u1"])
        var_total = var_fixed + var_random + float(s3["var_e"])
        if var_total > 0:
            r2_marginal = var_fixed / var_total
            r2_conditional = (var_fixed + var_random) / var_total
            gap = var_random / var_total
        else:
            r2_marginal = 0.0
            r2_conditional = 0.0
            gap = 0.0
        r2_rows.append({
            "response": resp_name,
            "test": "baseline_stage3",
            "r2_marginal": r2_marginal,
            "r2_conditional": r2_conditional,
            "gap": gap,
        })
    except Exception as e:
        r2_skipped.append((resp_name, "baseline_stage3", str(e)))

    if resp_name not in phase_a_results:
        continue

    for test_name, res in phase_a_results[resp_name].items():
        if test_name not in r2_moderator_order:
            continue
        if not str(res.get("status", "")).startswith("converged"):
            r2_skipped.append(
                (resp_name, test_name, f"status={res.get('status')}")
            )
            continue

        formula_str = res.get("formula", "")
        fe_dict = res.get("fe_params", {})
        if not formula_str or not fe_dict:
            r2_skipped.append((resp_name, test_name, "missing fe_params/formula"))
            continue

        try:
            _, x_mat = patsy.dmatrices(
                formula_str,
                df_model,
                return_type="dataframe",
                NA_action="drop",
            )
            if x_mat.shape[0] == 0:
                raise ValueError("Patsy returned 0 rows after NA drop")
            missing = set(x_mat.columns) - set(fe_dict.keys())
            if missing:
                raise ValueError(f"Missing fe_params for: {sorted(missing)}")
            beta = np.array(
                [fe_dict.get(col, 0.0) for col in x_mat.columns],
                dtype=float,
            )
            var_fixed = float(np.var(x_mat.values @ beta, ddof=0))
            var_random = float(res["var_u0"]) + float(res["var_u1"])
            var_total = var_fixed + var_random + float(res["var_e"])
            if var_total > 0:
                r2_marginal = var_fixed / var_total
                r2_conditional = (var_fixed + var_random) / var_total
                gap = var_random / var_total
            else:
                r2_marginal = 0.0
                r2_conditional = 0.0
                gap = 0.0
            r2_rows.append({
                "response": resp_name,
                "test": test_name,
                "r2_marginal": r2_marginal,
                "r2_conditional": r2_conditional,
                "gap": gap,
            })
        except Exception as e:
            r2_skipped.append((resp_name, test_name, str(e)))

df_r2_wavelet = pd.DataFrame(r2_rows)
df_r2_wavelet.to_csv("phase_a_r2_legacy_wavelet_gblup.csv", index=False)

panel_data = {}
for resp_name in wavelet_responses:
    baseline = df_r2_wavelet[
        (df_r2_wavelet["response"] == resp_name)
        & (df_r2_wavelet["test"] == "baseline_stage3")
    ]
    bars = []
    if len(baseline) > 0:
        row = baseline.iloc[0]
        bars.append((
            "baseline",
            {
                "r2m": float(row["r2_marginal"]) * 100,
                "gap": float(row["gap"]) * 100,
                "r2c": float(row["r2_conditional"]) * 100,
            },
        ))
    else:
        bars.append(("baseline", None))

    for test_name in r2_moderator_order:
        row = df_r2_wavelet[
            (df_r2_wavelet["response"] == resp_name)
            & (df_r2_wavelet["test"] == test_name)
        ]
        if len(row) == 0:
            bars.append((test_name, None))
        else:
            row = row.iloc[0]
            bars.append((
                test_name,
                {
                    "r2m": float(row["r2_marginal"]) * 100,
                    "gap": float(row["gap"]) * 100,
                    "r2c": float(row["r2_conditional"]) * 100,
                },
            ))
    panel_data[resp_name] = bars

valid_r2c = [
    d["r2c"]
    for resp_name in wavelet_responses
    for _, d in panel_data[resp_name]
    if d is not None
]

n_bars = 1 + len(r2_moderator_order)
x = np.arange(n_bars)
width = 0.65
ymax = max(valid_r2c) * 1.15

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)

for panel_idx, (resp_name, ax) in enumerate(zip(wavelet_responses, axes)):
    bars = panel_data[resp_name]

    for i, (_, d) in enumerate(bars):
        if d is None:
            continue

        is_base = i == 0
        c_fixed = c_fixed_dark if is_base else c_fixed_light
        c_gap = c_gap_dark if is_base else c_gap_light

        ax.bar(
            x[i],
            d["r2m"],
            width,
            color=c_fixed,
            edgecolor="white",
            linewidth=0.7,
            zorder=2,
        )
        ax.bar(
            x[i],
            d["gap"],
            width,
            bottom=d["r2m"],
            color=c_gap,
            edgecolor="white",
            linewidth=0.7,
            zorder=2,
        )

        ax.text(
            x[i],
            d["r2c"] + 0.4,
            f"{d['r2c']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#444444",
            fontweight="bold",
            zorder=4,
        )
        if d["r2m"] > 1.4:
            ax.text(
                x[i],
                d["r2m"] / 2,
                f"{d['r2m']:.1f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="white",
                fontweight="bold",
                zorder=3,
            )
        if d["gap"] > 2.0:
            ax.text(
                x[i],
                d["r2m"] + d["gap"] / 2,
                f"{d['gap']:.1f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white",
                fontweight="bold",
                zorder=3,
            )

    xticklabels = ["Base"] + r2_moderator_display
    ax.set_xticks(x)
    ax.set_xticklabels(
        xticklabels,
        rotation=40,
        ha="right",
        fontsize=8.5,
        color="#444444",
    )

    ax.set_title(
        wavelet_panel_titles[panel_idx],
        fontsize=12,
        fontweight="bold",
        loc="center",
        pad=10,
    )
    ax.set_ylim(0, ymax)
    if panel_idx == 0:
        ax.set_ylabel("% of Total Variance Explained", fontsize=10.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

legend_handles = [
    Patch(
        facecolor=c_fixed_dark,
        edgecolor="white",
        label=r"$R^2_{\mathrm{marginal}}$: fixed effects",
    ),
    Patch(
        facecolor=c_gap_dark,
        edgecolor="white",
        label="Personalization gap",
    ),
    Patch(
        facecolor=c_fixed_light,
        edgecolor="white",
        label=r"$R^2_{\mathrm{marginal}}$ + moderator",
    ),
    Patch(
        facecolor=c_gap_light,
        edgecolor="white",
        label="Gap after adding moderator",
    ),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=4,
    frameon=False,
    fontsize=8.5,
)

fig.suptitle(
    r"$R^2$ Decomposition: What You Ate vs Who You Are",
    fontsize=13,
    fontweight="bold",
    y=1.10,
)
plt.tight_layout()
plt.savefig(figure4c_output, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {figure4c_output}")
