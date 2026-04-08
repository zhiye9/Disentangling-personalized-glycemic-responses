"""
CGM data loading and preprocessing utilities.

This module provides tools to load, clean, and preprocess continuous glucose
monitoring (CGM) time-series data from multiple individuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class CGMProcessor:
    """Load and preprocess CGM data for population-level analysis.

    Parameters
    ----------
    glucose_col : str
        Column name for glucose values (mg/dL).
    time_col : str
        Column name for timestamps.
    subject_col : str
        Column name for subject/participant identifiers.
    low_cutoff : float
        Glucose values below this threshold are considered sensor errors
        and replaced with NaN (default 20 mg/dL).
    high_cutoff : float
        Glucose values above this threshold are considered sensor errors
        and replaced with NaN (default 600 mg/dL).
    """

    def __init__(
        self,
        glucose_col: str = "glucose",
        time_col: str = "timestamp",
        subject_col: str = "subject_id",
        low_cutoff: float = 20.0,
        high_cutoff: float = 600.0,
    ) -> None:
        self.glucose_col = glucose_col
        self.time_col = time_col
        self.subject_col = subject_col
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self._data: Optional[pd.DataFrame] = None

    def load(self, data: pd.DataFrame) -> "CGMProcessor":
        """Load a DataFrame containing CGM readings.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain columns for glucose values, timestamps, and subject IDs.

        Returns
        -------
        self : CGMProcessor
        """
        required = {self.glucose_col, self.time_col, self.subject_col}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = data.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df = df.sort_values([self.subject_col, self.time_col]).reset_index(drop=True)
        self._data = df
        return self

    def clean(self, interpolation_limit: int = 3) -> "CGMProcessor":
        """Remove physiologically implausible values and interpolate short gaps.

        Parameters
        ----------
        interpolation_limit : int
            Maximum number of consecutive missing readings to fill via linear
            interpolation (default 3, i.e. up to 15 min for 5-min sensors).

        Returns
        -------
        self : CGMProcessor
        """
        if self._data is None:
            raise RuntimeError("Call load() before clean().")

        df = self._data.copy()
        mask = (df[self.glucose_col] < self.low_cutoff) | (
            df[self.glucose_col] > self.high_cutoff
        )
        df.loc[mask, self.glucose_col] = np.nan

        # Per-subject linear interpolation of short gaps
        df[self.glucose_col] = df.groupby(self.subject_col)[
            self.glucose_col
        ].transform(
            lambda s: s.interpolate(method="linear", limit=interpolation_limit)
        )
        self._data = df
        return self

    def normalize(self, method: str = "z-score") -> "CGMProcessor":
        """Normalize glucose values per subject.

        Parameters
        ----------
        method : str
            Normalization method; one of ``'z-score'`` (subtract mean, divide
            by std) or ``'min-max'`` (scale to [0, 1]).

        Returns
        -------
        self : CGMProcessor
        """
        if self._data is None:
            raise RuntimeError("Call load() before normalize().")

        if method not in {"z-score", "min-max"}:
            raise ValueError("method must be 'z-score' or 'min-max'")

        df = self._data.copy()

        if method == "z-score":
            df["glucose_norm"] = df.groupby(self.subject_col)[
                self.glucose_col
            ].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
        else:  # min-max
            df["glucose_norm"] = df.groupby(self.subject_col)[
                self.glucose_col
            ].transform(lambda s: (s - s.min()) / (s.max() - s.min()))

        self._data = df
        return self

    def compute_summary_stats(self) -> pd.DataFrame:
        """Compute per-subject summary statistics.

        Returns
        -------
        pd.DataFrame
            One row per subject with columns: mean, std, cv (coefficient of
            variation), time_in_range (70–180 mg/dL fraction), gmi
            (Glucose Management Indicator).
        """
        if self._data is None:
            raise RuntimeError("Call load() before compute_summary_stats().")

        g = self.glucose_col

        def _stats(s: pd.Series) -> pd.Series:
            mean_ = s.mean()
            std_ = s.std(ddof=1)
            tir = ((s >= 70) & (s <= 180)).mean()
            gmi = 3.31 + 0.02392 * mean_  # mmol/mol to % conversion via ADA formula
            return pd.Series(
                {
                    "mean_glucose": mean_,
                    "std_glucose": std_,
                    "cv": std_ / mean_ if mean_ > 0 else np.nan,
                    "time_in_range": tir,
                    "gmi": gmi,
                    "n_readings": s.notna().sum(),
                }
            )

        return self._data.groupby(self.subject_col)[g].apply(_stats).reset_index()

    @property
    def data(self) -> pd.DataFrame:
        """Processed CGM DataFrame."""
        if self._data is None:
            raise RuntimeError("No data loaded. Call load() first.")
        return self._data

    @property
    def subjects(self) -> list:
        """List of unique subject identifiers."""
        return sorted(self.data[self.subject_col].unique().tolist())

    def simulate(
        self,
        n_subjects: int = 50,
        days: int = 14,
        seed: int = 42,
    ) -> "CGMProcessor":
        """Generate synthetic CGM data for testing and demonstration.

        Glucose dynamics are modelled as a sum of:
        * a stable fasting baseline per subject,
        * three meal-driven Gaussian peaks per day (breakfast, lunch, dinner),
        * subject-specific response amplitudes,
        * random Gaussian sensor noise.

        Parameters
        ----------
        n_subjects : int
            Number of simulated participants.
        days : int
            Simulation length in days.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        self : CGMProcessor
        """
        rng = np.random.default_rng(seed)
        records = []

        # Population-level meal-response amplitude and timing
        meal_times_h = [7.5, 12.5, 18.5]  # breakfast, lunch, dinner (hours of day)
        pop_amplitude = np.array([45.0, 40.0, 50.0])  # mg/dL peak above baseline
        pop_width = np.array([1.0, 1.0, 1.2])  # Gaussian sigma in hours

        # Per-subject parameters
        baselines = rng.normal(95, 15, n_subjects).clip(65, 140)
        amplitude_scale = rng.lognormal(0, 0.35, (n_subjects, 3))
        noise_std = rng.uniform(3, 8, n_subjects)

        start = pd.Timestamp("2024-01-01")
        freq_min = 5  # CGM reading every 5 minutes

        for i in range(n_subjects):
            subject_id = f"S{i+1:03d}"
            ts = pd.date_range(start, periods=days * 24 * 60 // freq_min, freq=f"{freq_min}min")
            t_hours = np.arange(len(ts)) * freq_min / 60.0  # elapsed hours

            glucose = np.full(len(ts), baselines[i])

            for j, (mt, pa, pw) in enumerate(
                zip(meal_times_h, pop_amplitude, pop_width)
            ):
                for day in range(days):
                    center = day * 24 + mt
                    glucose += (
                        pa
                        * amplitude_scale[i, j]
                        * np.exp(-0.5 * ((t_hours - center) / pw) ** 2)
                    )

            glucose += rng.normal(0, noise_std[i], len(ts))
            glucose = glucose.clip(40, 400)

            records.append(
                pd.DataFrame(
                    {
                        self.subject_col: subject_id,
                        self.time_col: ts,
                        self.glucose_col: glucose,
                    }
                )
            )

        sim_df = pd.concat(records, ignore_index=True)
        return self.load(sim_df)
