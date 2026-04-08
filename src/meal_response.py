"""
Postprandial glucose response extraction.

Given CGM data and meal event timestamps, this module extracts fixed-length
glucose trajectories around each meal (the postprandial curve) and computes
commonly used scalar features for downstream modelling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class MealResponseExtractor:
    """Extract postprandial glucose responses from CGM data.

    Parameters
    ----------
    glucose_col : str
        Column name for glucose values.
    time_col : str
        Column name for timestamps.
    subject_col : str
        Column name for subject identifiers.
    pre_meal_min : int
        Number of minutes before meal onset to include in the window.
    post_meal_min : int
        Number of minutes after meal onset to include in the window.
    cgm_freq_min : int
        CGM sampling frequency in minutes (default 5).
    baseline_window_min : int
        Duration (minutes) before meal used to estimate pre-meal baseline.
    """

    def __init__(
        self,
        glucose_col: str = "glucose",
        time_col: str = "timestamp",
        subject_col: str = "subject_id",
        pre_meal_min: int = 30,
        post_meal_min: int = 120,
        cgm_freq_min: int = 5,
        baseline_window_min: int = 30,
    ) -> None:
        self.glucose_col = glucose_col
        self.time_col = time_col
        self.subject_col = subject_col
        self.pre_meal_min = pre_meal_min
        self.post_meal_min = post_meal_min
        self.cgm_freq_min = cgm_freq_min
        self.baseline_window_min = baseline_window_min

        total_min = pre_meal_min + post_meal_min
        self.n_timepoints = total_min // cgm_freq_min + 1
        self.time_axis = np.arange(self.n_timepoints) * cgm_freq_min - pre_meal_min

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        cgm_data: pd.DataFrame,
        meal_events: pd.DataFrame,
        meal_time_col: str = "meal_time",
        meal_subject_col: Optional[str] = None,
        baseline_correct: bool = True,
    ) -> pd.DataFrame:
        """Extract postprandial glucose curves for each meal event.

        Parameters
        ----------
        cgm_data : pd.DataFrame
            Preprocessed CGM data.
        meal_events : pd.DataFrame
            Table of meal events with at least a timestamp column.
        meal_time_col : str
            Column name in meal_events for the meal timestamp.
        meal_subject_col : str, optional
            Column in meal_events for subject IDs.
        baseline_correct : bool
            If True, subtract each curve pre-meal mean glucose (excursion).

        Returns
        -------
        pd.DataFrame
            One row per meal with time-point columns and metadata.
        """
        if meal_subject_col is None:
            meal_subject_col = self.subject_col

        records = []
        for _, meal in meal_events.iterrows():
            subj = meal[meal_subject_col]
            mt = pd.Timestamp(meal[meal_time_col])
            curve, baseline = self._extract_single(cgm_data, subj, mt, baseline_correct)
            if curve is None:
                continue
            row = {
                self.subject_col: subj,
                "meal_time": mt,
                "baseline_glucose": baseline,
            }
            for k, v in zip(self._col_names(), curve):
                row[k] = v
            records.append(row)

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records).reset_index(drop=True)
        result["meal_index"] = result.groupby(self.subject_col).cumcount()
        return result

    def compute_features(self, response_df: pd.DataFrame) -> pd.DataFrame:
        """Compute scalar glycemic features from postprandial curves.

        Parameters
        ----------
        response_df : pd.DataFrame
            Output of extract().

        Returns
        -------
        pd.DataFrame
            One row per meal: peak_glucose, time_to_peak_min, iAUC,
            delta_return_min.
        """
        curve_cols = self._col_names()
        available = [c for c in curve_cols if c in response_df.columns]
        t = self.time_axis[: len(available)]

        records = []
        for _, row in response_df.iterrows():
            curve = row[available].to_numpy(dtype=float)
            peak_idx = int(np.nanargmax(curve))
            peak_g = float(curve[peak_idx])
            ttp = float(t[peak_idx])
            iauc = float(np.trapz(np.clip(curve, 0, None), t))

            post_mask = t >= 0
            post_curve = curve[post_mask]
            post_t = t[post_mask]
            returned = np.where(post_curve <= 10)[0]
            delta_return = float(post_t[returned[0]]) if len(returned) > 0 else np.nan

            records.append(
                {
                    self.subject_col: row[self.subject_col],
                    "meal_time": row["meal_time"],
                    "peak_glucose": peak_g,
                    "time_to_peak_min": ttp,
                    "iAUC": iauc,
                    "delta_return_min": delta_return,
                    "baseline_glucose": row.get("baseline_glucose", np.nan),
                }
            )

        return pd.DataFrame(records).reset_index(drop=True)

    def simulate_meal_events(
        self,
        cgm_data: pd.DataFrame,
        meal_hours: tuple = (7.5, 12.5, 18.5),
        seed: int = 0,
    ) -> pd.DataFrame:
        """Generate synthetic meal event timestamps aligned to CGM data.

        Parameters
        ----------
        cgm_data : pd.DataFrame
            CGM data with subject and time columns.
        meal_hours : tuple
            Hours of the day at which meals typically occur.
        seed : int
            Random seed for timing jitter.

        Returns
        -------
        pd.DataFrame
            Columns: subject_id, meal_time.
        """
        rng = np.random.default_rng(seed)
        records = []

        for subj, grp in cgm_data.groupby(self.subject_col):
            dates = grp[self.time_col].dt.normalize().unique()
            for d in dates:
                for mh in meal_hours:
                    jitter_min = int(rng.integers(-15, 15))
                    mt = pd.Timestamp(d) + pd.Timedelta(hours=mh, minutes=jitter_min)
                    records.append({self.subject_col: subj, "meal_time": mt})

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _col_names(self) -> list:
        names = []
        for t in self.time_axis:
            names.append(f"t{int(t):+d}" if t != 0 else "t0")
        return names

    def _extract_single(
        self,
        cgm_data: pd.DataFrame,
        subject,
        meal_time: pd.Timestamp,
        baseline_correct: bool,
    ):
        subj_data = cgm_data[cgm_data[self.subject_col] == subject].copy()
        window_start = meal_time - pd.Timedelta(minutes=self.pre_meal_min)
        window_end = meal_time + pd.Timedelta(minutes=self.post_meal_min)
        window = subj_data[
            (subj_data[self.time_col] >= window_start)
            & (subj_data[self.time_col] <= window_end)
        ].copy()

        if window.empty:
            return None, None

        window["t_rel"] = (
            window[self.time_col] - meal_time
        ).dt.total_seconds() / 60.0

        # Interpolate onto fixed grid
        grid_df = pd.DataFrame({"t_rel": self.time_axis})
        merged = pd.merge_asof(
            grid_df.sort_values("t_rel"),
            window[["t_rel", self.glucose_col]].sort_values("t_rel"),
            on="t_rel",
            tolerance=self.cgm_freq_min,
            direction="nearest",
        )
        curve = merged[self.glucose_col].to_numpy(dtype=float)

        baseline_mask = (merged["t_rel"] >= -self.baseline_window_min) & (
            merged["t_rel"] < 0
        )
        baseline_vals = curve[baseline_mask]
        baseline = float(np.nanmean(baseline_vals)) if len(baseline_vals) > 0 else np.nan

        if baseline_correct and not np.isnan(baseline):
            curve = curve - baseline

        return curve, baseline
