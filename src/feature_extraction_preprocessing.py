"""
Stationary Wavelet Transform (SWT) feature extraction and traditional glucose metrics from 3-hour postprandial CGM curves. This script takes a meal-level CSV file of postprandial glucose readings and computes 3 WT-derived feaures (frequency energies) and 3 traditional features (iAUC, peak rise, time to peak). Then merge features with person-level moderators, and preprocessing for modeling.

Input
-----
- df_meal_cgm : DataFrame with one row per meal and 13 CGM columns (glucose at 0, 15, 30, ..., 180 min).

Output
------
- df_model : preprocessed features and moderators. 
"""

import numpy as np
import pandas as pd
import pywt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# config
MOTHER_WAVELET = "db2"
SWT_LEVEL = 2
N_TIMEPOINTS = 13
GRID_STEP_MINUTES = 15
grid_minutes = [t * GRID_STEP_MINUTES for t in range(N_TIMEPOINTS)]
CGM_COLS = [f"pp_cgm_glucose_min_{m}" for m in grid_minutes]

# symmetric padding to the next multiple of 2^level (16)
PAD_LEFT = 1
PAD_RIGHT = 2
PAD_TARGET = PAD_LEFT + N_TIMEPOINTS + PAD_RIGHT  # 16

# Load the meal-level CGM table produced by preprocessing.
df_meal_cgm = pd.read_csv("df_cgm_meals.csv")

# baseline-corrected CGM matrix (n_meals, 13)
cgm_matrix = df_meal_cgm[CGM_COLS].values.astype(np.float64)
n_meals = cgm_matrix.shape[0]

cgm_matrix = cgm_matrix - cgm_matrix[:, [0]]

n_nan_rows = np.any(np.isnan(cgm_matrix), axis=1).sum()
if n_nan_rows > 0:
    raise ValueError(
        f"{n_nan_rows} rows contain NaN in CGM columns. "
    )

print(cgm_matrix.shape)


# symmetric padding for SWT
cgm_padded = np.pad(
    cgm_matrix,
    pad_width=((0, 0), (PAD_LEFT, PAD_RIGHT)),
    mode="symmetric",
)
print(cgm_padded.shape)

# SWT (db2, level 2), order: [cA_n, cD_n, ..., cD_1]. For level 2 this is [cA2, cD2, cD1].

coeffs_cA2 = np.zeros((n_meals, N_TIMEPOINTS))
coeffs_cD2 = np.zeros((n_meals, N_TIMEPOINTS))
coeffs_cD1 = np.zeros((n_meals, N_TIMEPOINTS))

for i in range(n_meals):
    sc = pywt.swt(cgm_padded[i, :], MOTHER_WAVELET, level=SWT_LEVEL,
                  trim_approx=True)
    cA2_full, cD2_full, cD1_full = sc[0], sc[1], sc[2]
    # Trim back to the 13 original timepoints. With PAD_LEFT=1, the original timepoints occupy padded positions [PAD_LEFT : PAD_LEFT + N_TIMEPOINTS].
    coeffs_cA2[i, :] = cA2_full[PAD_LEFT:PAD_LEFT + N_TIMEPOINTS]
    coeffs_cD2[i, :] = cD2_full[PAD_LEFT:PAD_LEFT + N_TIMEPOINTS]
    coeffs_cD1[i, :] = cD1_full[PAD_LEFT:PAD_LEFT + N_TIMEPOINTS]

    if (i + 1) % 10000 == 0:
        print(f"  Processed {i + 1}/{n_meals} meals")

# Attach the 39 per-timepoint coefficient columns (naming: swt_{wavelet}_{subfreq}_{timepoint_index}, t-index maps to minutes).
for t in range(N_TIMEPOINTS):
    df_meal_cgm[f"swt_{MOTHER_WAVELET}_cA2_{t}"] = coeffs_cA2[:, t]
    df_meal_cgm[f"swt_{MOTHER_WAVELET}_cD2_{t}"] = coeffs_cD2[:, t]
    df_meal_cgm[f"swt_{MOTHER_WAVELET}_cD1_{t}"] = coeffs_cD1[:, t]

# frequency energies computation (sum of squared  coefficients per frequency)
df_meal_cgm["swt_db2_energy_cA2"] = np.sum(coeffs_cA2 ** 2, axis=1)
df_meal_cgm["swt_db2_energy_cD2"] = np.sum(coeffs_cD2 ** 2, axis=1)
df_meal_cgm["swt_db2_energy_cD1"] = np.sum(coeffs_cD1 ** 2, axis=1)

for freq in ["cA2", "cD2", "cD1"]:
    col = f"swt_db2_energy_{freq}"
    vals = df_meal_cgm[col]

# traditional glucose metrics
times = np.asarray(grid_minutes, dtype=float)

# positive iAUC
cgm_matrix_pos = np.maximum(cgm_matrix, 0.0)
positive_iauc = np.trapezoid(cgm_matrix_pos, times, axis=1)
df_meal_cgm["iauc"] = positive_iauc     

#p eak glucose rise
peak_glucose_incr = np.max(cgm_matrix, axis=1)
df_meal_cgm["peak_rise"] = peak_glucose_incr

# peak time
peak_time_min = np.argmax(cgm_matrix, axis=1) * GRID_STEP_MINUTES
df_meal_cgm["peak_time"] = peak_time_min

# nutrient columns
CARB_COL = "available_carbohydrate_g"
PROTEIN_COL = "protein_g"
FAT_COL = "lipid_g"
FIBER_COL = "dietary_fiber_g"
ALCOHOL_COL = "alcohol_g"
NUTRIENT_COLS = [CARB_COL, PROTEIN_COL, FAT_COL, FIBER_COL]

# rsponse variables
RESPONSE_COLS = [
    "peak_rise", "iauc", "peak_time", "swt_db2_energy_cA2", "swt_db2_energy_cD2", "swt_db2_energy_cD1",
]

TRANSFORM_CONFIG = {
    "peak_rise": "log",
    "iauc": "log",
    "peak_time": "none", # not skewed
    "swt_db2_energy_cA2": "log",
    "swt_db2_energy_cD2": "log",
    "swt_db2_energy_cD1": "log",
}

# winsorization 
PHYSIO_BOUNDS = {
    "peak_rise": (-50.0, 250.0),     
    "iauc": (-3000.0, 20000.0),    
}
WAVELET_COLS = [
    "swt_db2_energy_cA2", "swt_db2_energy_cD2", "swt_db2_energy_cD1",
]
WAVELET_PCT_LOWER = 0.005   
WAVELET_PCT_UPPER = 0.995  

# meal timing
MEAL_TIME_COL = "meal_time"
MEAL_CATEGORY_COL = "meal_category"
MEAL_TIME_BINS = [0, 10, 14, 20, 24]
MEAL_TIME_LABELS = ["breakfast", "lunch", "dinner", "late_night"]

# person-level variables
PERSON_LEVEL_COLS = [
    "age",
    "sex",
    "bmi",
    "waist_to_hip_ratio",
    "total_scan_vat_mass",
    "body_comp_total_tissue_percent_fat",
    "bt__hba1c_float_value",
    "pco1", # microbiome
    "pco2",
    "gblup_cA2", # genetics
    "gblup_cD2",
    "gblup_cD1",
    "gblup_iAUC",
    "gblup_peak_rise",
    "gblup_peak_time",
]

# the processed moderator files to be merged with cgm metircs
PERSON_DATA_SOURCES = [
    ("anthropometrics", "df_anthropometrics.csv"),
    ("body_composition", "df_body.csv"),
    ("hba1c", "df_hba1c.csv"),
    ("genetics", "df_gblup.csv"),
    ("microbiome", "microbiome_genus_pcoa.csv"),
]

MIN_MEALS = 5
VIF_THRESHOLD = 5.0

# load data
df = df_meal_cgm.copy()
n_pids_start = df["participant_id"].nunique()
print(f"\n  Starting: {len(df)} meals, {n_pids_start} participants")

for label, source in PERSON_DATA_SOURCES:
    df_src = pd.read_csv(source)
    df_src = df_src.drop_duplicates(subset="participant_id", keep="first")
    new_cols = [c for c in df_src.columns
                if c != "participant_id" and c not in df.columns]
    if not new_cols:
        print(f"  [{label}] no new columns, skipped")
        continue
    n_before = len(df)
    df = df.merge(df_src[["participant_id"] + new_cols],
                  on="participant_id", how="left")
    pids_matched = df.loc[df[new_cols[0]].notna(), "participant_id"].nunique()
    print(f"  [{label}] +{len(new_cols)} cols, "
           f"{pids_matched}/{n_pids_start} participants matched")
    assert len(df) == n_before, "Row count changed after merge"

print(f"\n  Merged: {len(df)} meals, {len(df.columns)} columns")

# filter and detect available variables for modeling
available_responses = [r for r in RESPONSE_COLS if r in df.columns]
print(f"  Available responses: {available_responses}")

available_person_vars = {}
for col in PERSON_LEVEL_COLS:
    if col in df.columns and df[col].notna().any():
        available_person_vars[col] = col

df_model = df.dropna(subset=NUTRIENT_COLS + ["participant_id"]).copy()

if ALCOHOL_COL in df_model.columns:
    df_model["has_alcohol"] = (df_model[ALCOHOL_COL] > 0).astype(int)
else:
    df_model["has_alcohol"] = 0

meal_counts = df_model.groupby("participant_id").size()
valid_pids = meal_counts[meal_counts >= MIN_MEALS].index
df_model = df_model[df_model["participant_id"].isin(valid_pids)].copy()

print(f"\n  After filtering (>= {MIN_MEALS} meals per person): "
      f"{df_model['participant_id'].nunique()} participants, "
      f"{len(df_model)} meals")

# winsorization and log transformation
for resp, (lo, hi) in PHYSIO_BOUNDS.items():
    vals = df_model[resp].dropna()
    n_lo = int((vals < lo).sum())
    n_hi = int((vals > hi).sum())
    df_model[resp] = df_model[resp].clip(lower=lo, upper=hi)

# wavelet features (P0.5/P99.5).
for resp in WAVELET_COLS:
    vals = df_model[resp].dropna()
    lo = vals.quantile(WAVELET_PCT_LOWER)
    hi = vals.quantile(WAVELET_PCT_UPPER)
    n_lo = int((vals < lo).sum())
    n_hi = int((vals > hi).sum())
    df_model[resp] = df_model[resp].clip(lower=lo, upper=hi)

# peak_time is bounded by the measurement window and is not winsorized.

# log transform
response_runs = []
for resp in available_responses:
    transform = TRANSFORM_CONFIG.get(resp, "none")
    if transform == "log":
        log_col = f"{resp}_log"
        df_model[log_col] = np.log(df_model[resp])
        response_runs.append((log_col, f"{resp}_log", True))
        print(f"  {resp:30s}: log -> {log_col} "
              f"(raw skew={df_model[resp].skew():.2f})")
    else:
        response_runs.append((resp, f"{resp}_raw", False))
        print(f"  {resp:30s}: no transform")

# CWC decomposition of carbohydrate into within/between person.
CARB_SLOPE_VAR = "carb_within"

person_carb_mean = df_model.groupby("participant_id")[CARB_COL].transform("mean")

# within-person deviation, scaled by the SD of the pooled deviations.
df_model["carb_within_raw"] = df_model[CARB_COL] - person_carb_mean
CARB_WITHIN_SD = df_model["carb_within_raw"].std()
df_model["carb_within"] = df_model["carb_within_raw"] / CARB_WITHIN_SD

# between-person component: each person's mean carb, z-scored across persons.
person_means_unique = df_model.groupby("participant_id")[CARB_COL].mean()
CARB_BETWEEN_MEAN = person_means_unique.mean()
CARB_BETWEEN_SD = person_means_unique.std()
df_model["carb_between"] = (
    (person_carb_mean - CARB_BETWEEN_MEAN) / CARB_BETWEEN_SD
)

total_var = df_model[CARB_COL].var()
between_var = person_means_unique.var()
within_var = df_model["carb_within_raw"].var()

# fixed-effects formula
FORMULA_NUTRIENTS = f"{CARB_SLOPE_VAR} + carb_between + {PROTEIN_COL} + {FAT_COL} + {FIBER_COL} + has_alcohol"

# person-level variables encoding
person_ids = df_model["participant_id"].unique()
n_persons = len(person_ids)

sex_col = available_person_vars.get("sex")
sex_vals = df_model.groupby("participant_id")[sex_col].first()
print(f"\nSex variable '{sex_col}' value counts (person-level):")
print(sex_vals.value_counts().to_string())

unique_vals = df_model[sex_col].dropna().unique()
female_indicators = ["F", "Female", "female", "f", "woman", "Woman", "0"]
male_indicators = ["M", "Male", "male", "m", "man", "Man", "1"]
if set(unique_vals) & set(female_indicators):
    df_model["sex_binary"] = df_model[sex_col].map(
        {v: 0 for v in female_indicators}
        | {v: 1 for v in male_indicators}
    )

sex_dist = df_model.groupby("participant_id")["sex_binary"].first()
print(f"  Person-level: {(sex_dist == 0).sum()} male, "
        f"{(sex_dist == 1).sum()} female, ")

# z-score continuous person-level variables
zscore_vars = {}

continuous_keys = ["age", "bmi", "waist_to_hip_ratio", "body_comp_total_tissue_percent_fat", "total_scan_vat_mass", "bt__hba1c_float_value", "pco1", "pco2", "gblup_cA2", "gblup_cD2", "gblup_cD1", "gblup_iAUC", "gblup_peak_rise", "gblup_peak_time"]

for key in continuous_keys:
    col = available_person_vars.get(key)
    print(col)

    person_vals = df_model.groupby("participant_id")[col].first()
    p_mean = person_vals.mean()
    p_std = person_vals.std()
    n_valid = person_vals.notna().sum()
    n_missing = person_vals.isna().sum()

    if p_std == 0 or np.isnan(p_std):
        print(f"  {key}: std=0 or nan, SKIPPING")
        continue

    z_col = f"{key}_z"
    person_z = (person_vals - p_mean) / p_std
    df_model[z_col] = df_model["participant_id"].map(person_z)
    zscore_vars[key] = z_col

print(f"\n  Z-scored variables for modeling: {list(zscore_vars.keys())}")

# VIF diagnostics
vif_cols = [v for v in zscore_vars.values() if v in df_model.columns]
if "sex_binary" in df_model.columns:
    vif_cols.append("sex_binary")

df_person = df_model.drop_duplicates("participant_id")[vif_cols].dropna()
X = df_person.values
print(f"\n  VIF on {len(df_person)} participants, {len(vif_cols)} variables:")
for i, col in enumerate(vif_cols):
    vif_val = variance_inflation_factor(X, i)
    flag = " *** HIGH" if vif_val > VIF_THRESHOLD else ""
    print(f"    {col:35s} VIF = {vif_val:.2f}{flag}")