"""
Extract handcrafted and DWT-based features from OGTT glucose time series for the initial venous OGTT cohort.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks, welch


PROJECT_DIR = Path("~/Subtype_metabolic_classification/metabolic_subphenotypes_db_new")
INPUT_CSV = PROJECT_DIR / "all_cohort_metabolicsubphenotyping_ogtt_glucose_09102023.csv"
FEATURE_DIR = PROJECT_DIR / "results" / "dwt_frequency_features"

INITIAL_EXP_TYPE = "venous_without_matching_cgm_and_without_planned_athome_cgm"
VENOUS_METHOD = "CTRU_Venous"
SMOOTHING_SPAR = 0.35
EXPECTED_N_TIMEPOINTS = 16

DWT_LEVEL = 2
DWT_WAVELETS = ["db2", "db4", "sym2", "sym4", "coif1"]
DWT_COMPONENTS = ["cA2", "cD2", "cD1"]
DWT_METHODS = [
    "dwt_frequency_component_stats",
    "dwt_global_stats",
    "dwt_energy_entropy",
    "dwt_frequency_component_energy",
]
DWT_ABLATION_METHODS = [
    "dwt_frequency_component_stats",
    "dwt_global_stats",
    "dwt_energy_entropy",
]
DWT_COMPONENT_STATS = ["mean", "std", "energy", "entropy"]
DWT_COMPONENT_STAT_COLUMNS = [
    f"{component}_{stat}" for component in DWT_COMPONENTS for stat in DWT_COMPONENT_STATS
]
DWT_GLOBAL_COLUMNS = ["global_mean", "global_std", "global_energy", "global_entropy"]

def dwt_components(X: np.ndarray, wavelet: str, level: int = DWT_LEVEL) -> list[np.ndarray]:
    X_centered = X - X.mean(axis=1, keepdims=True)
    component_lists = [[] for _ in range(level + 1)]
    for sample in X_centered:
        coeffs = pywt.wavedec(sample, wavelet, level=level)
        for idx, coeff in enumerate(coeffs):
            component_lists[idx].append(coeff)
    return [np.array(values) for values in component_lists]


def dwt_feature_values(components: list[np.ndarray], method: str, component_names: list[str]) -> pd.DataFrame:
    if method == "dwt_frequency_component_stats":
        pieces = []
        for name, values in zip(component_names, components):
            abs_values = np.abs(values)
            row_sums = np.where(
                abs_values.sum(axis=1, keepdims=True) == 0,
                1,
                abs_values.sum(axis=1, keepdims=True),
            )
            probs = abs_values / row_sums
            pieces.append(
                pd.DataFrame(
                    {
                        f"{name}_mean": np.mean(values, axis=1),
                        f"{name}_std": np.std(values, axis=1),
                        f"{name}_energy": np.sum(values ** 2, axis=1),
                        f"{name}_entropy": -np.sum(probs * np.log2(probs + 1e-12), axis=1),
                    }
                )
            )
        return pd.concat(pieces, axis=1)

    if method == "dwt_frequency_component_energy":
        return pd.DataFrame(
            {f"{name}_energy": np.sum(values ** 2, axis=1) for name, values in zip(component_names, components)}
        )

    values = np.hstack(components)
    abs_values = np.abs(values)
    row_sums = np.where(
        abs_values.sum(axis=1, keepdims=True) == 0,
        1,
        abs_values.sum(axis=1, keepdims=True),
    )
    probs = abs_values / row_sums
    global_stats = pd.DataFrame(
        {
            "global_mean": np.mean(values, axis=1),
            "global_std": np.std(values, axis=1),
            "global_energy": np.sum(values ** 2, axis=1),
            "global_entropy": -np.sum(probs * np.log2(probs + 1e-12), axis=1),
        }
    )
    if method == "dwt_global_stats":
        return global_stats
    if method == "dwt_energy_entropy":
        return global_stats[["global_energy", "global_entropy"]]
    raise ValueError(f"Unknown DWT method: {method}")


# preprocess OGTT data
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(INPUT_CSV)
venous_ogtt = raw[
    raw["sample_location_extraction_method"].eq(VENOUS_METHOD)
    & raw["exp_type"].eq(INITIAL_EXP_TYPE)
].copy()

ogtt_wide = venous_ogtt.pivot_table(
    index="timepoint",
    columns="subject_id",
    values="glucose",
    aggfunc="mean",
).sort_index()
ogtt_wide.index = pd.to_numeric(ogtt_wide.index, errors="coerce")
ogtt_wide = ogtt_wide.sort_index()

ogtt_imputed = pd.DataFrame(index=ogtt_wide.index)
for subject_id, series in ogtt_wide.items():
    ogtt_imputed[subject_id] = series.interpolate(method="linear", limit_direction="both")

ogtt_normalized = pd.DataFrame(index=ogtt_imputed.index)
for subject_id, series in ogtt_imputed.items():
    ogtt_normalized[subject_id] = (series - series.mean()) / series.std(ddof=1)

ogtt_timepoints = ogtt_normalized.index.to_numpy(dtype=float)
ogtt_smoothed = pd.DataFrame(index=ogtt_normalized.index)
for subject_id, series in ogtt_normalized.items():
    spline = UnivariateSpline(
        x=ogtt_timepoints,
        y=series,
        s=SMOOTHING_SPAR * len(ogtt_timepoints),
    )
    ogtt_smoothed[subject_id] = spline(ogtt_timepoints)

# extract handcrafted features
ogtt_for_features = ogtt_imputed.copy()
ogtt_for_features.index = pd.to_numeric(ogtt_for_features.index, errors="coerce")
ogtt_for_features = ogtt_for_features.sort_index().iloc[1:, :]
ogtt_auc_df = ogtt_for_features.loc[(ogtt_for_features.index >= 0.0) & (ogtt_for_features.index <= 180.0)]
ogtt_auc_time = ogtt_auc_df.index.to_numpy(dtype=float)
ogtt_auc_values = ogtt_auc_df.to_numpy(dtype=float)
ogtt_baseline = ogtt_for_features.loc[ogtt_for_features.index == 0.0].iloc[0].to_numpy(dtype=float)

ogtt_auc = np.trapezoid(ogtt_auc_values, x=ogtt_auc_time, axis=0)
ogtt_iauc = ogtt_auc - (180.0 * ogtt_baseline)
ogtt_positive_values = ogtt_auc_values - ogtt_baseline
ogtt_positive_values[ogtt_positive_values < 0.0] = 0.0
ogtt_pauc = np.trapezoid(ogtt_positive_values, x=ogtt_auc_time, axis=0)
ogtt_nauc = ogtt_pauc - ogtt_iauc

ogtt_time = ogtt_for_features.index.to_numpy(dtype=float)
ogtt_baseline_idx = ogtt_for_features.index.get_loc(0.0)
ogtt_baseline_time = float(ogtt_for_features.index[ogtt_baseline_idx])
ogtt_last_time = ogtt_time[-1]
ogtt_time_to_peak = []
ogtt_time_peak_to_baseline = []
ogtt_slope_baseline_peak = []
ogtt_slope_peak_last = []
ogtt_below_baseline_flag = []
ogtt_cv = []

for _, series in ogtt_for_features.items():
    values = series.to_numpy(dtype=float)
    idx_max = int(np.nanargmax(values))
    peak_time = ogtt_time[idx_max]
    peak_value = values[idx_max]
    baseline_value = values[ogtt_baseline_idx]
    post_peak_mask = ogtt_time > peak_time
    below_baseline_mask = values < baseline_value
    candidates = ogtt_time[post_peak_mask & below_baseline_mask]
    mean = series.mean()
    std = series.std(ddof=1)

    ogtt_time_to_peak.append(peak_time)
    ogtt_time_peak_to_baseline.append(np.nan if candidates.size == 0 else candidates[0] - peak_time)
    ogtt_slope_baseline_peak.append(
        np.nan if peak_time == ogtt_baseline_time else (peak_value - baseline_value) / (peak_time - ogtt_baseline_time)
    )
    ogtt_slope_peak_last.append(
        np.nan if ogtt_last_time == peak_time else (values[-1] - peak_value) / (ogtt_last_time - peak_time)
    )
    ogtt_below_baseline_flag.append(bool(np.any(post_peak_mask & below_baseline_mask)))
    ogtt_cv.append(np.nan if mean == 0 or np.isnan(mean) or np.isnan(std) else (std / mean) * 100.0)

handcrafted_ogtt_features = pd.DataFrame(
    {
        "subject_id": ogtt_for_features.columns.astype(str),
        "ogtt_fpg": ogtt_baseline,
        "ogtt_60": ogtt_for_features.loc[ogtt_for_features.index == 60.0].iloc[0].to_numpy(dtype=float),
        "ogtt_120": ogtt_for_features.loc[ogtt_for_features.index == 120.0].iloc[0].to_numpy(dtype=float),
        "ogtt_180": ogtt_for_features.loc[ogtt_for_features.index == 180.0].iloc[0].to_numpy(dtype=float),
        "ogtt_auc": ogtt_auc,
        "ogtt_iauc": ogtt_iauc,
        "ogtt_pauc": ogtt_pauc,
        "ogtt_nauc": ogtt_nauc,
        "ogtt_max": ogtt_for_features.max(axis=0).to_numpy(dtype=float),
        "ogtt_curve_size": np.sum(np.abs(np.diff(ogtt_for_features.to_numpy(dtype=float), axis=0)), axis=0),
        "ogtt_cv": ogtt_cv,
        "ogtt_time_baseline_peak": ogtt_time_to_peak,
        "ogtt_time_peak_baseline": ogtt_time_peak_to_baseline,
        "ogtt_slope_baseline_peak": ogtt_slope_baseline_peak,
        "ogtt_slope_peak_last": ogtt_slope_peak_last,
        "ogtt_time_below_basline": ogtt_below_baseline_flag,
    }
)
handcrafted_ogtt_features["ogtt_time_peak_baseline"] = handcrafted_ogtt_features["ogtt_time_peak_baseline"].fillna(180)
handcrafted_ogtt_features["ogtt_time_below_basline"] = handcrafted_ogtt_features["ogtt_time_below_basline"].astype(int)
handcrafted_ogtt_features.to_csv(FEATURE_DIR / "handcrafted_ogtt_features.csv", index=False)


# DWT features
subject_ids = ogtt_smoothed.columns.astype(str).to_numpy()
X_smoothed = ogtt_smoothed.T.to_numpy(dtype=float)
dwt_full_rows = []
for wavelet in DWT_WAVELETS:
    components = dwt_components(X_smoothed, wavelet=wavelet)
    for method in DWT_METHODS:
        values = dwt_feature_values(components, method, DWT_COMPONENTS)
        values.insert(0, "level", DWT_LEVEL)
        values.insert(0, "wavelet", wavelet)
        values.insert(0, "feature_method", method)
        values.insert(0, "extraction", f"{method}_{wavelet}")
        values.insert(0, "subject_id", subject_ids)
        dwt_full_rows.append(values)

dwt_full_features = pd.concat(dwt_full_rows, ignore_index=True, sort=False)
dwt_full_metadata = ["subject_id", "extraction", "feature_method", "wavelet", "level"]
dwt_full_columns = dwt_full_metadata + [
    col for col in DWT_COMPONENT_STAT_COLUMNS + DWT_GLOBAL_COLUMNS if col in dwt_full_features.columns
]
dwt_full_features = dwt_full_features.reindex(columns=dwt_full_columns)
dwt_full_features.to_csv(FEATURE_DIR / "dwt_full_features_level2.csv", index=False)


# DWT per-frequency ablation features
dwt_db2_components = dwt_components(X_smoothed, wavelet="db2")
dwt_frequency_ablation_rows = []
for removed_component in DWT_COMPONENTS:
    kept_components = [values for name, values in zip(DWT_COMPONENTS, dwt_db2_components) if name != removed_component]
    kept_component_names = [name for name in DWT_COMPONENTS if name != removed_component]
    for method in DWT_ABLATION_METHODS:
        values = dwt_feature_values(kept_components, method, kept_component_names)
        values.insert(0, "removed_frequency_component", removed_component)
        values.insert(0, "level", DWT_LEVEL)
        values.insert(0, "wavelet", "db2")
        values.insert(0, "feature_method", method)
        values.insert(0, "extraction", f"{method}_db2_without_{removed_component}")
        values.insert(0, "subject_id", subject_ids)
        dwt_frequency_ablation_rows.append(values)

dwt_frequency_ablation_features = pd.concat(dwt_frequency_ablation_rows, ignore_index=True, sort=False)
dwt_frequency_ablation_metadata = [
    "subject_id",
    "extraction",
    "feature_method",
    "wavelet",
    "level",
    "removed_frequency_component",
]
dwt_frequency_ablation_columns = dwt_frequency_ablation_metadata + [
    col for col in DWT_COMPONENT_STAT_COLUMNS + DWT_GLOBAL_COLUMNS if col in dwt_frequency_ablation_features.columns
]
dwt_frequency_ablation_features = dwt_frequency_ablation_features.reindex(columns=dwt_frequency_ablation_columns)
dwt_frequency_ablation_features.to_csv(FEATURE_DIR / "dwt_frequency_ablation_features_level2.csv", index=False)


# FFT/Welch frequency features
X_frequency = X_smoothed - X_smoothed.mean(axis=1, keepdims=True)
fft_frequencies = np.arange(EXPECTED_N_TIMEPOINTS // 2 + 1) / EXPECTED_N_TIMEPOINTS
fft_welch_rows = []
for subject_id, sample in zip(subject_ids, X_frequency):
    magnitudes = np.abs(np.fft.rfft(sample))
    non_dc = magnitudes[1:]
    non_dc_freq = fft_frequencies[1:]
    non_dc_total = float(np.sum(non_dc))
    if not np.isfinite(non_dc_total) or non_dc_total <= 0:
        fft_max_amplitude = 0.0
        fft_dominant_frequency = np.nan
        fft75_frequency = np.nan
    else:
        max_idx = int(np.argmax(non_dc))
        fft_max_amplitude = float(non_dc[max_idx])
        fft_dominant_frequency = float(non_dc_freq[max_idx]) if fft_max_amplitude > 0 else np.nan
        if fft_max_amplitude > 0:
            cumulative = np.cumsum(non_dc)
            idx75 = int(np.searchsorted(cumulative, 0.75 * non_dc_total, side="left"))
            fft75_frequency = float(non_dc_freq[min(idx75, len(non_dc_freq) - 1)])
        else:
            fft75_frequency = np.nan

    welch_freq, psd = welch(
        sample,
        fs=1.0,
        nperseg=8,
        noverlap=4,
        window="hann",
        scaling="density",
        return_onesided=True,
        detrend=False,
    )
    non_dc_psd = psd[1:]
    if not np.isfinite(float(np.sum(non_dc_psd))) or float(np.sum(non_dc_psd)) <= 0:
        psd_max_amplitude = 0.0
    else:
        psd_max_amplitude = float(psd[int(np.argmax(non_dc_psd) + 1)])

    peak_indices, _ = find_peaks(non_dc)
    fft_welch_rows.append(
        {
            "subject_id": subject_id,
            "fft_max_amplitude": fft_max_amplitude,
            "fft_dominant_frequency": fft_dominant_frequency,
            "fft75_frequency": fft75_frequency,
            "psd_max_amplitude": psd_max_amplitude,
            "fft_local_peak_count": int(len(peak_indices)),
            "welch_n_frequency_bins": int(len(welch_freq)),
        }
    )

fft_welch_features = pd.DataFrame(fft_welch_rows)
fft_welch_features.to_csv(FEATURE_DIR / "fft_welch_features.csv", index=False)

print(f"Feature CSVs saved to: {FEATURE_DIR}")
