"""
Probabilistic metabolic phenotyping via Monte Carlo sampling from the posterior distribution of random intercept and carbohydrate's random slope BLUPs. Monte Carlo draws are classified into four phenotypes based on the sign of the two BLUPs, and the resulting phenotype probabilities are linked to person-level cardiometabolic markers and pre-diabetes risk.

Phenotype definitions (threshold = 0):
  Q0: u0 < 0, u1 <= 0  -> Low magnitude, low sensitivity
  Q1: u0 <= 0, u1 > 0  -> Low magnitude, high sensitivity
  Q2: u0 > 0, u1 <= 0  -> High magnitude, low sensitivity   
  Q3: u0 > 0, u1 > 0   -> High magnitude, high sensitivity

This script should be run after progressive_modeling.py.
"""

import json
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.legend_handler import HandlerTuple
from scipy import stats
from statsmodels.stats.multitest import multipletests


PHENOTYPE_RESPONSE = "swt_db2_energy_cA2_log"

N_MC_SAMPLES = 10000
RANDOM_SEED = 42
P_CONFIDENT = 0.60
P_TENTATIVE = 0.50
PREDIAB_LOWER = 5.7
DIAB_UPPER = 6.5

PID_COL = "participant_id"
PHENO_NAMES = [
    "Low magnitude, low sensitivity",
    "Low magnitude, high sensitivity",
    "High magnitude, low sensitivity",
    "High magnitude, high sensitivity",
]
PHENO_SHORT = ["LL", "LH", "HL", "HH"]
P_COLS = ["P_LL", "P_LH", "P_HL", "P_HH"]

VALIDATION_MARKERS = [
    ("hba1c", "HbA1c (%)", ["age_c", "sex_binary"], False),
    ("bmi", "BMI (kg/m2)", ["age_c", "sex_binary"], False),
    ("waist_to_hip_ratio", "Waist-to-hip ratio", ["age_c", "sex_binary"], False),
    ("body_fat_pct", "Total body fat (%)", ["age_c", "sex_binary"], False),
    ("vat_mass", "Visceral adipose mass (g)", ["age_c", "sex_binary"], False),
]

PERSON_COL_MAP = {
    "age": "age",
    "sex": "sex",
    "bmi": "bmi",
    "waist_to_hip_ratio": "waist_to_hip_ratio",
    "body_fat_pct": "body_comp_total_tissue_percent_fat",
    "vat_mass": "total_scan_vat_mass",
    "hba1c": "bt__hba1c_float_value",
}


# load cA2 random-slope fit
model_info = stage3_baseline[PHENOTYPE_RESPONSE]
fit_cA2 = model_info["fit"]

var_u0 = float(model_info.get("var_u0", fit_cA2.cov_re.iloc[0, 0]))
var_u1 = float(model_info.get("var_u1", fit_cA2.cov_re.iloc[1, 1]))
var_e = float(model_info.get("var_e", fit_cA2.scale))
cor_u01 = float(model_info.get("cor_u01", np.nan))
gamma10 = float(fit_cA2.fe_params.get(CARB_SLOPE_VAR, 0.0))
tau0 = np.sqrt(max(var_u0, 0))
tau1 = np.sqrt(max(var_u1, 0))


# extract BLUP posterior
blups_raw = fit_cA2.random_effects
cov_re_raw = fit_cA2.random_effects_cov
participant_ids = list(blups_raw.keys())
n_participants = len(participant_ids)

blup_array = np.zeros((n_participants, 2))
cov_array = np.zeros((n_participants, 2, 2))

for i, pid in enumerate(participant_ids):
    b = blups_raw[pid]
    c = cov_re_raw[pid]
    b_values = b.values if hasattr(b, "values") else np.asarray(b)
    c_values = c.values if hasattr(c, "values") else np.asarray(c)
    blup_array[i, :] = b_values[:2]
    cov_array[i, :, :] = c_values[:2, :2]

se_intercept = np.sqrt(np.maximum(cov_array[:, 0, 0], 0))
se_slope = np.sqrt(np.maximum(cov_array[:, 1, 1], 0))

# monte carlo classification
rng = np.random.default_rng(RANDOM_SEED)
posterior_probs = np.zeros((n_participants, 4))

for i in range(n_participants):
    mean_i = blup_array[i]
    cov_i = cov_array[i]
    try:
        samples = rng.multivariate_normal(mean_i, cov_i, size=N_MC_SAMPLES)
    except np.linalg.LinAlgError:
        samples = np.column_stack([
            rng.normal(mean_i[0], np.sqrt(max(cov_i[0, 0], 0)), N_MC_SAMPLES),
            rng.normal(mean_i[1], np.sqrt(max(cov_i[1, 1], 0)), N_MC_SAMPLES),
        ])

    u0_s = samples[:, 0]
    u1_s = samples[:, 1]
    posterior_probs[i, 0] = ((u0_s < 0) & (u1_s <= 0)).mean()
    posterior_probs[i, 1] = ((u0_s <= 0) & (u1_s > 0)).mean()
    posterior_probs[i, 2] = ((u0_s > 0) & (u1_s <= 0)).mean()
    posterior_probs[i, 3] = ((u0_s > 0) & (u1_s > 0)).mean()

p_max = posterior_probs.max(axis=1)
assigned = posterior_probs.argmax(axis=1)
confident = p_max >= P_CONFIDENT
tentative = (p_max >= P_TENTATIVE) & (p_max < P_CONFIDENT)
uncertain = p_max < P_TENTATIVE

for q, name in enumerate(PHENO_NAMES):
    n_q = int((assigned == q).sum())

print("\n  Confidence tiers:")
print(f"    Confident (P >= {P_CONFIDENT}): "
      f"{confident.sum()} ({confident.mean() * 100:.1f}%)")
print(f"    Tentative ({P_TENTATIVE}-{P_CONFIDENT}): "
      f"{tentative.sum()} ({tentative.mean() * 100:.1f}%)")
print(f"    Uncertain (< {P_TENTATIVE}): "
      f"{uncertain.sum()} ({uncertain.mean() * 100:.1f}%)")
print(f"    Median P_max: {np.median(p_max):.3f}")


# probability-weighted validation
method_a_results = {}
contrast_rows = []
contrasts = [
    ("HH vs LL", 3, 0),
    ("HL vs LL", 2, 0),
    ("LH vs LL", 1, 0),
    ("HH vs HL", 3, 2),
]

for marker, label, covariates, is_binary in VALIDATION_MARKERS:
    needed_cols = [marker] + P_COLS + [c for c in covariates if c in df_val.columns]
    df_cc = df_val[needed_cols].dropna()

    X_parts = [df_cc[P_COLS].values]
    col_names = list(PHENO_SHORT)
    for cov in covariates:
        if cov in df_cc.columns:
            X_parts.append(df_cc[[cov]].values)
            col_names.append(cov)
    X = np.hstack(X_parts)
    y = df_cc[marker].values.astype(float)

    model = sm.OLS(y, X).fit().get_robustcov_results(cov_type="HC3")
    ci = model.conf_int()

    alphas = {PHENO_SHORT[k]: float(model.params[k]) for k in range(4)}
    alpha_se = {PHENO_SHORT[k]: float(model.bse[k]) for k in range(4)}
    contrast_results = {}

    for cname, idx_a, idx_b in contrasts:
        c_vec = np.zeros(len(model.params))
        c_vec[idx_a] = 1
        c_vec[idx_b] = -1
        t_res = model.t_test(c_vec)
        diff = float(np.asarray(t_res.effect).squeeze())
        se = float(np.asarray(t_res.sd).squeeze())
        t_stat = float(np.asarray(t_res.tvalue).squeeze())
        p_val = float(np.asarray(t_res.pvalue).squeeze())
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {cname:<23s} {diff:>+9.3f} {se:>9.3f} "
              f"{t_stat:>8.2f} {p_val:>11.2e} {sig}")
        contrast_results[cname] = {
            "diff": diff,
            "se": se,
            "t": t_stat,
            "p": p_val,
        }
        contrast_rows.append({
            "marker": marker,
            "label": label,
            "contrast": cname,
            "diff": diff,
            "se": se,
            "t": t_stat,
            "p": p_val,
            "n": len(df_cc),
        })

    method_a_results[marker] = {
        "n": len(df_cc),
        "label": label,
        "alphas": alphas,
        "alpha_se": alpha_se,
        "contrasts": contrast_results,
        "model": model,
    }

contrast_df = pd.DataFrame(contrast_rows)
if len(contrast_df) > 0:
    _, p_fdr, _, _ = multipletests(contrast_df["p"].values, method="fdr_bh")
    contrast_df["p_fdr"] = p_fdr
    contrast_df["fdr_significant"] = contrast_df["p_fdr"] < 0.05
    contrast_df.to_csv("phenotype_marker_contrasts.csv", index=False)


# confident-only sensitivity analysis
df_conf = df_val[df_val["confident"]].copy()
df_rh = df_conf[df_conf["assigned"].isin([0, 3])].copy()

method_b_results = {}

for marker, label, _, is_binary in VALIDATION_MARKERS:
    if marker not in df_rh.columns:
        continue
    df_test = df_rh[[marker, "assigned"]].dropna()
    if len(df_test) < 20:
        continue

    grp_res = df_test.loc[df_test["assigned"] == 0, marker]
    grp_hr = df_test.loc[df_test["assigned"] == 3, marker]
    if len(grp_res) == 0 or len(grp_hr) == 0:
        continue

    if is_binary:
        table = np.array([
            [(grp_res == 1).sum(), (grp_res == 0).sum()],
            [(grp_hr == 1).sum(), (grp_hr == 0).sum()],
        ])
        if table.min() > 0:
            test_stat, p_val, _, _ = stats.chi2_contingency(table)
        else:
            test_stat, p_val = np.nan, np.nan
        diff = grp_hr.mean() - grp_res.mean()
        method_b_results[marker] = {
            "label": label,
            "n_res": len(grp_res),
            "n_hr": len(grp_hr),
            "mean_res": grp_res.mean(),
            "mean_hr": grp_hr.mean(),
            "diff": diff,
            "test_stat": test_stat,
            "p": p_val,
            "test_type": "chi2",
        }
    else:
        test_stat, p_val = stats.ttest_ind(grp_hr, grp_res, equal_var=False)
        diff = grp_hr.mean() - grp_res.mean()
        pooled_sd = np.sqrt((grp_res.var() + grp_hr.var()) / 2)
        cohens_d = diff / pooled_sd if pooled_sd > 0 else np.nan
        method_b_results[marker] = {
            "label": label,
            "n_res": len(grp_res),
            "n_hr": len(grp_hr),
            "mean_res": grp_res.mean(),
            "mean_hr": grp_hr.mean(),
            "diff": diff,
            "test_stat": test_stat,
            "p": p_val,
            "cohens_d": cohens_d,
            "test_type": "welch_t",
        }

    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"  {label:<28s}: diff={diff:+.3f}, p={p_val:.2e} {sig}")


# consistency comparison
summary_rows = []
direction_count = 0
strong_count = 0
total_count = 0

for marker, label, _, _ in VALIDATION_MARKERS:
    a = method_a_results.get(marker)
    b = method_b_results.get(marker)


    if a is not None:
        a_contrast = a["contrasts"].get("HighRisk vs Res", {})
        a_diff = a_contrast.get("diff", np.nan)
        a_p = a_contrast.get("p", np.nan)
        a_n = a["n"]
        a_str = f"N={a_n}, diff={a_diff:+.3f}, p={a_p:.1e}"
    else:
        a_diff = np.nan
        a_p = np.nan
        a_n = np.nan
        a_str = "N/A"

    if b is not None:
        b_diff = b["diff"]
        b_p = b["p"]
        b_n = b["n_res"] + b["n_hr"]
        b_str = f"N={b_n}, diff={b_diff:+.3f}, p={b_p:.1e}"
    else:
        b_diff = np.nan
        b_p = np.nan
        b_n = np.nan
        b_str = "N/A"

    if not (np.isnan(a_diff) or np.isnan(b_diff)):
        total_count += 1
        same_direction = np.sign(a_diff) == np.sign(b_diff)
        a_sig = a_p < 0.05
        b_sig = b_p < 0.05
        if same_direction:
            direction_count += 1
        if same_direction and a_sig and b_sig:
            evidence = "Strongly consistent"
            strong_count += 1
        elif same_direction and (a_sig or b_sig):
            evidence = "Directionally supportive"
        elif same_direction:
            evidence = "Same direction, weak"
        else:
            evidence = "Potential conflict"
    else:
        same_direction = np.nan
        a_sig = False
        b_sig = False
        evidence = "N/A"

    summary_rows.append({
        "marker": marker,
        "label": label,
        "method_a_n": a_n,
        "method_a_diff": a_diff,
        "method_a_p": a_p,
        "method_b_n": b_n,
        "method_b_diff": b_diff,
        "method_b_p": b_p,
        "same_direction": same_direction,
        "method_a_significant": a_sig,
        "method_b_significant": b_sig,
        "evidence_label": evidence,
    })

summary_comparison_df = pd.DataFrame(summary_rows)
summary_comparison_df.to_csv(
    "method_a_vs_method_b_directional_consistency.csv",
    index=False,
)


# pre-diabetes validation
prediab_results = {}
PREDIAB_CSV = "prediabetes_swt.csv"

df_prediab = pd.read_csv(PREDIAB_CSV)
prediab_ids = set(df_prediab[PID_COL].dropna())
df_val["is_prediabetic"] = df_val[PID_COL].isin(prediab_ids).astype(int)

df_pre = df_val[df_val["is_prediabetic"] == 1].copy()
df_non = df_val[df_val["is_prediabetic"] == 0].copy()

df_val["prediab_hba1c_group"] = np.nan
pre_mask = df_val["is_prediabetic"] == 1
df_val.loc[pre_mask & df_val["hba1c"].isna(), "prediab_hba1c_group"] = (
    "prediab_no_hba1c"
)
df_val.loc[pre_mask & (df_val["hba1c"] < PREDIAB_LOWER),
            "prediab_hba1c_group"] = "hba1c_normal"
df_val.loc[
    pre_mask
    & (df_val["hba1c"] >= PREDIAB_LOWER)
    & (df_val["hba1c"] < DIAB_UPPER),
    "prediab_hba1c_group",
] = "hba1c_concordant"
df_val.loc[pre_mask & (df_val["hba1c"] >= DIAB_UPPER),
            "prediab_hba1c_group"] = "hba1c_elevated"

contingency = np.zeros((4, 2), dtype=int)
for q, name in enumerate(PHENO_NAMES):
    n_pre = int((df_pre["assigned"] == q).sum())
    n_non = int((df_non["assigned"] == q).sum())
    contingency[q, :] = [n_pre, n_non]
    pct_pre = n_pre / len(df_pre) * 100 if len(df_pre) else 0
    pct_non = n_non / len(df_non) * 100 if len(df_non) else 0
    print(f"    {name:<25s}: pre={pct_pre:>5.1f}% "
            f"non-pre={pct_non:>5.1f}%")
chi2_quad, p_chi2_quad, dof_quad, _ = stats.chi2_contingency(contingency)

for q, name in enumerate(PHENO_NAMES):
    col = P_COLS[q]

cov_cols = []
if "age_c" in df_val.columns and df_val["age_c"].notna().mean() > 0.5:
    cov_cols.append("age_c")
if "sex_binary" in df_val.columns and df_val["sex_binary"].notna().mean() > 0.5:
    cov_cols.append("sex_binary")

needed_cols = ["is_prediabetic"] + P_COLS + cov_cols
df_cc = df_val[needed_cols].dropna()
X_parts = [df_cc[P_COLS].values]
col_names = list(PHENO_SHORT)
for cov in cov_cols:
    X_parts.append(df_cc[[cov]].values)
    col_names.append(cov)
X = np.hstack(X_parts)
y = df_cc["is_prediabetic"].values.astype(float)

model_prediab = sm.OLS(y, X).fit().get_robustcov_results(cov_type="HC3")
print("\n  Method A probability-weighted pre-diabetes prevalence:")
ci = model_prediab.conf_int()

prediab_contrast_rows = []
for cname, idx_a, idx_b in [
    ("HighRisk vs Resilient", 3, 0),
    ("GlobHigh vs Resilient", 2, 0),
    ("CarbSens vs Resilient", 1, 0),
    ("HighRisk vs GlobHigh", 3, 2),
]:
    c_vec = np.zeros(len(model_prediab.params))
    c_vec[idx_a] = 1
    c_vec[idx_b] = -1
    t_res = model_prediab.t_test(c_vec)
    diff = float(np.asarray(t_res.effect).squeeze())
    se = float(np.asarray(t_res.sd).squeeze())
    p_val = float(np.asarray(t_res.pvalue).squeeze())
    prediab_contrast_rows.append({
        "contrast": cname,
        "diff": diff,
        "se": se,
        "p": p_val,
    })

model_prediab_logit = None
try:
    model_prediab_logit = sm.Logit(y, X).fit(disp=0, maxiter=200)
    print(f"\n  Logistic phenotype model: pseudo R2={model_prediab_logit.prsquared:.4f}")
    for cname, idx_a, idx_b in [
        ("HH vs LL", 3, 0),
        ("HL vs LL", 2, 0),
        ("LH vs LL", 1, 0),
        ("HH vs HL", 3, 2),
    ]:
        c_vec = np.zeros(len(model_prediab_logit.params))
        c_vec[idx_a] = 1
        c_vec[idx_b] = -1
        t_res = model_prediab_logit.t_test(c_vec)
        log_or = float(np.asarray(t_res.effect).squeeze())
        se = float(np.asarray(t_res.sd).squeeze())
        p_val = float(np.asarray(t_res.pvalue).squeeze())
        or_val = np.exp(log_or)
        or_lo = np.exp(log_or - 1.96 * se)
        or_hi = np.exp(log_or + 1.96 * se)
        print(f"    {cname:<25s}: OR={or_val:.3f} "
                f"[{or_lo:.3f}, {or_hi:.3f}], p={p_val:.2e}")
except Exception as e:
    print(f"\n  Logistic phenotype model failed: {e}")

df_conf = df_val[df_val["confident"]].copy()
df_res = df_conf[df_conf["assigned"] == 0]
df_hrs = df_conf[df_conf["assigned"] == 3]
if len(df_res) > 0 and len(df_hrs) > 0:
    table = np.array([
        [
            int(df_res["is_prediabetic"].sum()),
            int((df_res["is_prediabetic"] == 0).sum()),
        ],
        [
            int(df_hrs["is_prediabetic"].sum()),
            int((df_hrs["is_prediabetic"] == 0).sum()),
        ],
    ])
    or_fisher, p_fisher = stats.fisher_exact(table)
    chi2_b, p_chi2_b, _, _ = stats.chi2_contingency(table)
    prev_res = df_res["is_prediabetic"].mean()
    prev_hrs = df_hrs["is_prediabetic"].mean()

blup_logit_results = {}
for label, x_cols in [
    ("u0_z + u1_z + covariates", ["u0_z", "u1_z"] + cov_cols),
    ("u0_z only + covariates", ["u0_z"] + cov_cols),
    ("u1_z only + covariates", ["u1_z"] + cov_cols),
]:
    needed_cols = ["is_prediabetic"] + x_cols
    df_cc_blup = df_val[needed_cols].dropna()
    if len(df_cc_blup) < 30 or df_cc_blup["is_prediabetic"].nunique() < 2:
        continue
    X_blup = sm.add_constant(df_cc_blup[x_cols])
    y_blup = df_cc_blup["is_prediabetic"].values.astype(float)
    try:
        res = sm.Logit(y_blup, X_blup).fit(disp=0, maxiter=200)
    except Exception as e:
        print(f"  BLUP logistic model failed ({label}): {e}")
        continue
    blup_logit_results[label] = res

    for term in ["u0_z", "u1_z"]:
        if term in x_cols:
            idx = ["const"] + x_cols
            j = idx.index(term)
            print(f"    {term}: OR={np.exp(res.params[j]):.3f}, "
                    f"p={res.pvalues[j]:.2e}")

    if (
        "u0_z + u1_z + covariates" in blup_logit_results
        and "u0_z only + covariates" in blup_logit_results
    ):
        res_full = blup_logit_results["u0_z + u1_z + covariates"]
        res_u0 = blup_logit_results["u0_z only + covariates"]
        lr_stat = -2 * (res_u0.llf - res_full.llf)
        lr_p = stats.chi2.sf(lr_stat, df=1)

    prediab_results = {
        "n_prediab": int(df_val["is_prediabetic"].sum()),
        "n_total": len(df_val),
        "method_a_contrasts": prediab_contrast_rows,
    }
else:
    print(f"  {PREDIAB_CSV} not found; skipping pre-diabetes validation.")


# plot Figure 5a
phenotype_colors = ["#39A432", "#7A52A6", "#428DBF", "#E2071E"]

fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

h_unc = ax.scatter(
    blup_array[uncertain, 0],
    blup_array[uncertain, 1],
    c="lightgray",
    alpha=0.2,
    s=6,
)
h_ten = ax.scatter(
    blup_array[tentative, 0],
    blup_array[tentative, 1],
    c="gold",
    alpha=0.3,
    s=10,
)

for q_idx in range(4):
    mask = (assigned == q_idx) & confident
    if mask.sum() > 0:
        ax.scatter(
            blup_array[mask, 0],
            blup_array[mask, 1],
            c=phenotype_colors[q_idx],
            alpha=0.7,
            s=20,
        )

h_conf = tuple(
    ax.scatter([], [], c=phenotype_colors[i], s=20, alpha=0.7)
    for i in range(4)
)
ax.legend(
    [h_unc, h_ten, h_conf],
    [
        f"Uncertain ({int(uncertain.sum())})",
        f"Tentative ({int(tentative.sum())})",
        f"Confident ({int(confident.sum())})",
    ],
    handler_map={tuple: HandlerTuple(ndivide=None, pad=1)},
    fontsize=9,
    loc="upper left",
    framealpha=0.9,
    bbox_to_anchor=(0.0, 0.88),
)

ax.axhline(0, color="gray", ls="--", lw=0.8)
ax.axvline(0, color="gray", ls="--", lw=0.8)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
offset_x = (xlim[1] - xlim[0]) * 0.03
offset_y = (ylim[1] - ylim[0]) * 0.03
q_counts = [(assigned == i).sum() for i in range(4)]
annotations = [
    (f"Low magnitude\nLow sensitivity\n(n = {q_counts[0]})",
     phenotype_colors[0], xlim[0] + offset_x, ylim[0] + offset_y,
     "left", "bottom"),
    (f"Low magnitude\nHigh sensitivity\n(n = {q_counts[1]})",
     phenotype_colors[1], xlim[0] + offset_x, ylim[1] - offset_y,
     "left", "top"),
    (f"High magnitude\nLow sensitivity\n(n = {q_counts[2]})",
     phenotype_colors[2], xlim[1] - offset_x, ylim[0] + offset_y,
     "right", "bottom"),
    (f"High magnitude\nHigh sensitivity\n(n = {q_counts[3]})",
     phenotype_colors[3], xlim[1] - offset_x, ylim[1] - offset_y,
     "right", "top"),
]
for text, color, x, y, ha, va in annotations:
    ax.text(
        x,
        y,
        text,
        fontsize=9,
        color=color,
        fontweight="bold",
        ha=ha,
        va=va,
        alpha=0.85,
    )

ax.set_xlabel("BLUP intercept (u0)")
ax.set_ylabel("BLUP carb slope (u1)")
ax.set_title("Response Distribution in BLUP Space")
plt.tight_layout()
plt.savefig("plot6_blup_space_p_06.png", dpi=600, bbox_inches="tight")
plt.show()
print("Saved: plot6_blup_space_p_06.png")


# plot Figure 5b
df_pre_plot = df_val[df_val["is_prediabetic"] == 1].copy()
df_non_plot = df_val[df_val["is_prediabetic"] == 0].copy()

fig, ax = plt.subplots(figsize=(8, 7))
fig.suptitle(
    "Pre-diabetes Validation (cA2)",
    fontsize=14,
    fontweight="bold",
)
ax.scatter(
    df_non_plot["u0_blup"],
    df_non_plot["u1_blup"],
    alpha=0.06,
    s=6,
    c="grey",
    rasterized=True,
    label="Non-pre-diabetic",
)
ax.scatter(
    df_pre_plot["u0_blup"],
    df_pre_plot["u1_blup"],
    alpha=0.8,
    s=30,
    c="red",
    edgecolors="darkred",
    linewidths=0.5,
    label="Pre-diabetic",
    zorder=5,
)
ax.axhline(0, color="black", lw=0.5, ls="--")
ax.axvline(0, color="black", lw=0.5, ls="--")
ax.set_xlabel("u0 (intercept)")
ax.set_ylabel("u1 (carb slope)")
ax.set_title("BLUP Space")
ax.legend(fontsize=6, loc="upper left")
plt.tight_layout()
plt.savefig("plot1_blup_scatter_prediab-pmax_06.png",
            dpi=300, bbox_inches="tight")
plt.show()
print("Saved: plot1_blup_scatter_prediab-pmax_06.png")

# plot Figure 5c
dotplot_colors = ["#4CAF50", "#8E6BA5", "#5DA5CB", "#D94F4F"]
dotplot_legend_labels = [
    "Low magnitude\nLow sensitivity",
    "Low magnitude\nHigh sensitivity",
    "High magnitude\nLow sensitivity",
    "High magnitude\nHigh sensitivity",
]
display_config = {
    "body_fat_pct": ("Total body fat", "Total body fat (%)", 100),
    "total_body_fat": ("Total body fat", "Total body fat (%)", 100),
    "body_fat_percentage": ("Total body fat", "Total body fat (%)", 100),
    "hba1c": ("HbA1c", "HbA1c (%)", 1),
    "bmi": ("BMI", r"BMI (kg m$^{-2}$)", 1),
    "waist_to_hip_ratio": (
        "Waist-to-hip ratio",
        "Waist-to-hip ratio",
        1,
    ),
    "vat_mass": ("Visceral adipose mass", "Visceral adipose mass (g)", 1),
    "visceral_adipose": (
        "Visceral adipose mass",
        "Visceral adipose mass (g)",
        1,
    ),
}
dot_linewidth = 1.2
errorbar_lw = 1.8
capsize = 5

all_pairs = list(combinations(range(4), 2))
pairwise_results = {}
for marker, label, _, is_binary in VALIDATION_MARKERS:
    if is_binary or marker not in method_a_results:
        continue

    model = method_a_results[marker]["model"]
    pw = {}
    for idx_a, idx_b in all_pairs:
        c_vec = np.zeros(len(model.params))
        c_vec[idx_a] = 1
        c_vec[idx_b] = -1
        t_res = model.t_test(c_vec)
        p_val = float(np.asarray(t_res.pvalue).squeeze())
        diff = float(np.asarray(t_res.effect).squeeze())
        pw[(idx_a, idx_b)] = {"p": p_val, "p_raw": p_val, "diff": diff}

    pairwise_results[marker] = pw

pw_rows = []
for marker, pw in pairwise_results.items():
    for (idx_a, idx_b), result in pw.items():
        pw_rows.append({
            "marker": marker,
            "idx_a": idx_a,
            "idx_b": idx_b,
            "p_raw": result["p_raw"],
        })

if len(pw_rows) > 0:
    pw_df = pd.DataFrame(pw_rows)
    _, pw_df["p_fdr"], _, _ = multipletests(
        pw_df["p_raw"].values,
        method="fdr_bh",
    )
    for _, row in pw_df.iterrows():
        pair_key = (int(row["idx_a"]), int(row["idx_b"]))
        pairwise_results[row["marker"]][pair_key]["p_fdr"] = row["p_fdr"]
    print(
        f"BH-FDR: {len(pw_df)} tests, "
        f"{(pw_df['p_raw'] < 0.05).sum()} raw sig -> "
        f"{(pw_df['p_fdr'] < 0.05).sum()} FDR sig"
    )

continuous_markers = [
    (marker, label)
    for marker, label, _, is_binary in VALIDATION_MARKERS
    if not is_binary and marker in method_a_results
]
n_plots = len(continuous_markers)

n_cols = min(n_plots, 5)
n_rows = (n_plots + n_cols - 1) // n_cols
fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(3.0 * n_cols, 5.0 * n_rows),
    squeeze=False,
)
axes = axes.flatten()
x_pos = np.arange(4)

for idx, (marker, label) in enumerate(continuous_markers):
    ax = axes[idx]
    res = method_a_results[marker]
    alphas = res["alphas"]
    alpha_ses = res["alpha_se"]

    short_title = label
    ylabel = label
    scale = 1
    marker_lower = marker.lower()
    for key, (title_val, ylabel_val, scale_val) in display_config.items():
        if key in marker_lower or marker_lower in key:
            short_title = title_val
            ylabel = ylabel_val
            scale = scale_val
            break

    means = np.array([alphas[s] for s in PHENO_SHORT]) * scale
    ses = np.array([alpha_ses[s] for s in PHENO_SHORT]) * scale
    ci_96 = 1.96 * ses

    for q_idx in range(4):
        ax.errorbar(
            x_pos[q_idx],
            means[q_idx],
            yerr=ci_96[q_idx],
            fmt="o",
            color=dotplot_colors[q_idx],
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=dot_linewidth,
            ecolor=dotplot_colors[q_idx],
            elinewidth=errorbar_lw,
            capsize=capsize,
            capthick=errorbar_lw,
            zorder=5,
        )

    ax.set_xticks([])
    ax.set_title(
        f"{short_title}, n = {res['n']:,}",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax.set_ylabel(ylabel, fontsize=10)

    y_min_data = (means - ci_96).min()
    y_max_data = (means + ci_96).max()
    y_range = y_max_data - y_min_data
    if not np.isfinite(y_range) or y_range <= 0:
        y_range = max(abs(y_max_data), 1.0) * 0.1
    y_bottom = y_min_data - 0.15 * y_range
    y_top = y_max_data + 0.55 * y_range
    ax.set_ylim(y_bottom, y_top)

    pw = pairwise_results.get(marker, {})
    sig_pairs = []
    for (idx_a, idx_b), result in pw.items():
        p_val = result.get("p_fdr", np.nan)
        if p_val < 0.05:
            sig_text = (
                "***" if p_val < 0.001
                else "**" if p_val < 0.01
                else "*"
            )
            sig_pairs.append((idx_a, idx_b, p_val, sig_text))
    sig_pairs.sort(key=lambda x: (abs(x[1] - x[0]), x[0]))

    bracket_y = y_max_data + 0.08 * y_range
    bracket_step = 0.08 * y_range
    bracket_h = 0.02 * y_range
    for pair_idx, (idx_a, idx_b, _, sig_text) in enumerate(sig_pairs):
        y_bracket = bracket_y + pair_idx * bracket_step
        ax.plot(
            [x_pos[idx_a], x_pos[idx_a], x_pos[idx_b], x_pos[idx_b]],
            [y_bracket, y_bracket + bracket_h,
                y_bracket + bracket_h, y_bracket],
            lw=1.0,
            color="black",
        )
        ax.text(
            (x_pos[idx_a] + x_pos[idx_b]) / 2,
            y_bracket + bracket_h,
            sig_text,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    if len(sig_pairs) > 0:
        final_bracket_y = bracket_y + len(sig_pairs) * bracket_step
        if final_bracket_y + bracket_h * 2 > y_top:
            ax.set_ylim(y_bottom, final_bracket_y + bracket_h * 3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for idx in range(n_plots, len(axes)):
    axes[idx].set_visible(False)

legend_handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=1.2,
        markersize=10,
    )
    for color in dotplot_colors
]
fig.legend(
    legend_handles,
    dotplot_legend_labels,
    loc="lower center",
    ncol=4,
    fontsize=10,
    bbox_to_anchor=(0.5, -0.10),
    columnspacing=2.0,
    handletextpad=0.8,
)

plt.tight_layout()
plt.savefig("validation_method_a_dotplot_v4_fdr.png",
            dpi=300, bbox_inches="tight")
plt.show()