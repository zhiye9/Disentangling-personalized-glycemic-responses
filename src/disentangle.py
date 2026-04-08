"""
Core algorithms for disentangling personalized vs. population-level
glycemic responses to meals.

Two complementary approaches are implemented:

1. **LinearMixedEffectsDisentangler** - uses a linear mixed-effects model
   (via statsmodels) to decompose each scalar glycemic feature (e.g. iAUC)
   into a fixed population-level component and subject-level random effects.

2. **MatrixFactorizationDisentangler** - decomposes the (subjects x timepoints)
   matrix of mean postprandial curves into a low-rank population component
   (shared basis) plus subject-specific coefficients using Non-negative Matrix
   Factorization (NMF) or truncated SVD.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import StandardScaler


class LinearMixedEffectsDisentangler:
    """Disentangle glycemic responses using a linear mixed-effects model.

    For each scalar feature (e.g. iAUC, peak_glucose) the model is::

        y_ij = beta_0 + u_i + epsilon_ij

    where y_ij is the feature value for subject i at meal j,
    beta_0 is the population intercept (fixed effect), u_i is the
    subject-level random intercept, and epsilon_ij is residual noise.

    When meal-level covariates (e.g. carbohydrate content) are provided they
    are included as additional fixed effects.

    Parameters
    ----------
    feature_cols : list of str
        Scalar feature columns to model.
    subject_col : str
        Column identifying subjects.
    covariate_cols : list of str, optional
        Additional fixed-effect covariates (e.g. meal macronutrients).
    """

    def __init__(
        self,
        feature_cols: list,
        subject_col: str = "subject_id",
        covariate_cols: Optional[list] = None,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.subject_col = subject_col
        self.covariate_cols = list(covariate_cols) if covariate_cols else []
        self._results: dict = {}
        self._random_effects: dict = {}

    def fit(self, features_df: pd.DataFrame) -> "LinearMixedEffectsDisentangler":
        """Fit mixed-effects models for each scalar feature.

        Parameters
        ----------
        features_df : pd.DataFrame
            Output of MealResponseExtractor.compute_features(), optionally
            merged with meal covariate columns.

        Returns
        -------
        self
        """
        try:
            import statsmodels.formula.api as smf
        except ImportError as exc:
            raise ImportError("statsmodels is required. pip install statsmodels") from exc

        for feat in self.feature_cols:
            df = features_df[[self.subject_col, feat] + self.covariate_cols].dropna()
            if df.empty or df[self.subject_col].nunique() < 2:
                warnings.warn(f"Skipping {feat}: insufficient data.")
                continue

            fixed_terms = " + ".join(self.covariate_cols) if self.covariate_cols else "1"
            formula = f"{feat} ~ {fixed_terms}"
            re_formula = "~1"

            try:
                model = smf.mixedlm(
                    formula,
                    df,
                    groups=df[self.subject_col],
                    re_formula=re_formula,
                )
                result = model.fit(reml=True, method="lbfgs", disp=False)
                self._results[feat] = result
                re_df = result.random_effects
                self._random_effects[feat] = pd.DataFrame(
                    {
                        self.subject_col: list(re_df.keys()),
                        f"{feat}_random_effect": [v.iloc[0] for v in re_df.values()],
                    }
                )
            except Exception as exc:
                warnings.warn(f"Model fit failed for {feat}: {exc}")

        return self

    def get_population_effects(self) -> pd.DataFrame:
        """Return population-level (fixed) effect estimates.

        Returns
        -------
        pd.DataFrame
            Columns: feature, term, estimate, std_err, p_value.
        """
        rows = []
        for feat, res in self._results.items():
            for term in res.params.index:
                rows.append(
                    {
                        "feature": feat,
                        "term": term,
                        "estimate": res.params[term],
                        "std_err": res.bse[term],
                        "p_value": res.pvalues[term],
                    }
                )
        return pd.DataFrame(rows)

    def get_subject_effects(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Return per-subject random intercepts (personalized component).

        Parameters
        ----------
        features_df : pd.DataFrame
            Same data used for fitting.

        Returns
        -------
        pd.DataFrame
            One row per subject; columns are subject_id plus one random-effect
            column per feature.
        """
        subjects = pd.DataFrame(
            {self.subject_col: features_df[self.subject_col].unique()}
        )
        result = subjects.copy()
        for feat, re_df in self._random_effects.items():
            result = result.merge(re_df, on=self.subject_col, how="left")
        return result

    @property
    def fitted_models(self) -> dict:
        """Dictionary mapping feature name to fitted MixedLM result."""
        return self._results


class MatrixFactorizationDisentangler:
    """Disentangle glycemic response curves via matrix factorization.

    The (n_subjects, n_timepoints) matrix of mean postprandial glucose curves
    is factorized as::

        X ≈ W * H

    where H (shape k x T) captures k latent population-level
    trajectory shapes and W (shape N x k) contains per-subject loadings
    (the personalized component).

    Parameters
    ----------
    n_components : int
        Number of latent components.
    method : str
        Factorization algorithm: 'nmf' (Non-negative MF) or 'svd'
        (truncated SVD / PCA).
    subject_col : str
        Column identifying subjects.
    """

    def __init__(
        self,
        n_components: int = 3,
        method: str = "svd",
        subject_col: str = "subject_id",
    ) -> None:
        if method not in {"nmf", "svd"}:
            raise ValueError("method must be 'nmf' or 'svd'")
        self.n_components = n_components
        self.method = method
        self.subject_col = subject_col

        self._W: Optional[np.ndarray] = None  # subject loadings (N x k)
        self._H: Optional[np.ndarray] = None  # latent trajectories (k x T)
        self._subject_index: Optional[list] = None
        self._time_axis: Optional[np.ndarray] = None
        self._scaler: Optional[StandardScaler] = None

    def fit(
        self,
        response_df: pd.DataFrame,
        time_axis: np.ndarray,
    ) -> "MatrixFactorizationDisentangler":
        """Fit the factorization model on averaged postprandial curves.

        Parameters
        ----------
        response_df : pd.DataFrame
            Output of MealResponseExtractor.extract() - one row per meal.
        time_axis : np.ndarray
            Time axis (minutes relative to meal) corresponding to curve columns.

        Returns
        -------
        self
        """
        time_cols = [c for c in response_df.columns if c.startswith("t") and c not in ("t_rel",)]
        # Keep only numeric time-point columns
        time_cols = [c for c in time_cols if c != "meal_time" and c != "meal_index"]

        # Average curves per subject
        mean_curves = response_df.groupby(self.subject_col)[time_cols].mean()
        X = mean_curves.values.astype(float)

        self._subject_index = list(mean_curves.index)
        self._time_axis = time_axis

        # Handle NaNs by column-mean imputation
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        self._scaler = StandardScaler(with_std=False)
        X_centered = self._scaler.fit_transform(X)

        if self.method == "nmf":
            # Shift so all values non-negative
            X_nn = X_centered - X_centered.min()
            decomp = NMF(
                n_components=self.n_components,
                init="nndsvda",
                random_state=0,
                max_iter=500,
            )
            self._W = decomp.fit_transform(X_nn)
            self._H = decomp.components_
        else:
            decomp = TruncatedSVD(n_components=self.n_components, random_state=0)
            self._W = decomp.fit_transform(X_centered)
            self._H = decomp.components_

        return self

    def population_trajectories(self) -> pd.DataFrame:
        """Return the latent population-level response trajectories.

        Returns
        -------
        pd.DataFrame
            Shape (n_components, n_timepoints) with time axis as column names.
        """
        if self._H is None:
            raise RuntimeError("Call fit() first.")
        cols = [f"t{int(t):+d}" if t != 0 else "t0" for t in self._time_axis]
        return pd.DataFrame(
            self._H, columns=cols[: self._H.shape[1]],
            index=[f"component_{i+1}" for i in range(self.n_components)]
        )

    def subject_loadings(self) -> pd.DataFrame:
        """Return per-subject loadings on each latent component.

        Returns
        -------
        pd.DataFrame
            One row per subject with columns: subject_id, component_1, ...,
            component_k.
        """
        if self._W is None:
            raise RuntimeError("Call fit() first.")
        df = pd.DataFrame(
            self._W,
            columns=[f"component_{i+1}" for i in range(self.n_components)],
        )
        df.insert(0, self.subject_col, self._subject_index)
        return df

    def reconstruct(self) -> pd.DataFrame:
        """Reconstruct the mean postprandial curves from the factorization.

        Returns
        -------
        pd.DataFrame
            Same shape as subject-averaged input; index is subject IDs.
        """
        if self._W is None or self._H is None:
            raise RuntimeError("Call fit() first.")
        X_hat = self._W @ self._H
        X_hat = self._scaler.inverse_transform(X_hat)
        cols = [f"t{int(t):+d}" if t != 0 else "t0" for t in self._time_axis]
        df = pd.DataFrame(
            X_hat,
            index=self._subject_index,
            columns=cols[: X_hat.shape[1]],
        )
        df.index.name = self.subject_col
        return df.reset_index()

    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of total variance explained by each component (SVD only).

        Returns
        -------
        np.ndarray or None
            Array of length n_components, or None if method is NMF.
        """
        if self.method != "svd" or self._W is None:
            return None
        total_var = np.sum(self._W ** 2) + np.sum(self._H ** 2)
        per_comp = np.array([
            self._W[:, i].var() + self._H[i, :].var()
            for i in range(self.n_components)
        ])
        total = per_comp.sum()
        return per_comp / total if total > 0 else per_comp


class GlycemicResponseDisentangler:
    """High-level facade combining both disentangling approaches.

    This class orchestrates the full pipeline:
    1. Fit a linear mixed-effects model on scalar features.
    2. Fit a matrix factorization on curve shapes.

    Parameters
    ----------
    n_components : int
        Number of latent trajectory components for matrix factorization.
    factorization_method : str
        'svd' or 'nmf'.
    subject_col : str
        Column name for subject identifiers.
    """

    def __init__(
        self,
        n_components: int = 3,
        factorization_method: str = "svd",
        subject_col: str = "subject_id",
    ) -> None:
        self.n_components = n_components
        self.factorization_method = factorization_method
        self.subject_col = subject_col

        self.lme_model = LinearMixedEffectsDisentangler(
            feature_cols=["iAUC", "peak_glucose", "time_to_peak_min"],
            subject_col=subject_col,
        )
        self.mf_model = MatrixFactorizationDisentangler(
            n_components=n_components,
            method=factorization_method,
            subject_col=subject_col,
        )

    def fit(
        self,
        response_df: pd.DataFrame,
        features_df: pd.DataFrame,
        time_axis: np.ndarray,
    ) -> "GlycemicResponseDisentangler":
        """Fit both disentangling models.

        Parameters
        ----------
        response_df : pd.DataFrame
            Output of MealResponseExtractor.extract().
        features_df : pd.DataFrame
            Output of MealResponseExtractor.compute_features().
        time_axis : np.ndarray
            Time axis (minutes) for curve columns in response_df.

        Returns
        -------
        self
        """
        self.lme_model.fit(features_df)
        self.mf_model.fit(response_df, time_axis)
        return self

    def population_summary(self) -> pd.DataFrame:
        """Fixed-effect estimates from the mixed-effects model."""
        return self.lme_model.get_population_effects()

    def subject_summary(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Per-subject personalized components (LME random effects + loadings).

        Parameters
        ----------
        features_df : pd.DataFrame
            Features data used in fitting.

        Returns
        -------
        pd.DataFrame
            One row per subject combining LME random effects and MF loadings.
        """
        lme_effects = self.lme_model.get_subject_effects(features_df)
        mf_loadings = self.mf_model.subject_loadings()
        return lme_effects.merge(mf_loadings, on=self.subject_col, how="outer")
