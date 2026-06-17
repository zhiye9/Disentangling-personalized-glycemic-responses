"""
Stationary Wavelet Transform (SWT) feature extraction and traditional glucose metrics from 3-hour postprandial CGM curves. This script takes a meal-level CSV file of postprandial glucose readings and computes 3 WT-derived feaures (frequency energies) and 3 traditional features (iAUC, peak rise, time to peak).

Input
-----
- df_meal_cgm : DataFrame with one row per meal and 13 CGM columns (glucose at 0, 15, 30, ..., 180 min).

Output
------
- df_swt : df_meal_cgm with appended columns:
    swt_db2_energy_cA2, swt_db2_energy_cD2, swt_db2_energy_cD1 (WT-derived features), iauc, peak_rise, peak_time (traditional glucose metrics).
"""

import numpy as np
import pandas as pd
import pywt

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

df_swt = df_meal_cgm.copy()