"""
Nested-CV stratified classification using extracted features and DWT features.
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path("/home/zhi/data/GP_diet_challenge/Subtype_metabolic_classification/metabolic_subphenotypes_db_new")
FEATURE_DIR = PROJECT_DIR / "results" / "dwt_frequency_features"
CV_DIR = PROJECT_DIR / "results" / "dwt_frequency_cv" / "ml"
PLOT_DIR = CV_DIR / "plots"

ANNOTATION_CSV = PROJECT_DIR / "All_cohort_MuscleIR_BetaCellFunction_Annotation.csv"
INITIAL_PHENOTYPES_CSV = PROJECT_DIR / "initial_cohort_metabolic_phenotypes.csv"
INITIAL_EXP_TYPE = "venous_without_matching_cgm_and_without_planned_athome_cgm"

N_ITERATIONS = 100
N_FOLDS = 5
RANDOM_SEED = 10

TARGETS = {
    "muscle_ir": {"label_col": "sspg_2_classes", "pos_label": "IR", "source": "annotation"},
    "beta_cell": {"label_col": "di_2_classes_median", "pos_label": "Dysfunction", "source": "annotation"},
    "incretin_effect": {"label_col": "IE_Class", "pos_label": "Dysfunction", "source": "initial_phenotypes"},
    "hepatic_ir": {"label_col": "HepaticIR_Class", "pos_label": "IR", "source": "initial_phenotypes"},
}
TARGET_ORDER = ["muscle_ir", "beta_cell", "incretin_effect", "hepatic_ir"]
TARGET_LABELS = {
    "muscle_ir": "Muscle IR",
    "beta_cell": "Beta-Cell\nFunction",
    "incretin_effect": "Incretin\nEffect",
    "hepatic_ir": "Hepatic IR",
}

PARAMETERS_GRID = {
    "C": [0.01, 0.1, 1, 10, 100],
    "n_estimators": [20, 50, 100, 150, 200],
    "learning_rate": [0.001, 0.01, 0.1, 0.5, 1.0],
}
CSET_CV = {
    "L1_logistic": [0.01, 0.1, 1, 10, 100],
    "L2_Logistic": [0.01, 0.1, 1, 10, 100],
    "RBF_SVC": [0.01, 0.1, 1, 10, 100],
}
PARAIND_CV = [0, 0, 0, 1]
CLASSIFIER_ORDER = ["L1_logistic", "L2_Logistic", "RBF_SVC", "RFR"]

METADATA_COLUMNS = {
    "subject_id",
    "extraction",
    "feature_method",
    "wavelet",
    "level",
    "removed_frequency_component",
}
RESULT_COLUMNS = [
    "Features",
    "Extraction",
    "Iteration",
    "Fold",
    "Classifier",
    "hyperparameter",
    "auROC",
    "Accuracy",
    "Recall",
    "Precision",
    "F1",
]

FREQUENCY_COMPONENT_ORDER = ["cA2", "cD2", "cD1"]
FREQUENCY_COMPONENT_LABELS = {
    "cA2": "low-frequency cA2",
    "cD2": "detail-frequency cD2",
    "cD1": "detail-frequency cD1",
}
FREQUENCY_COMPONENT_COLORS = {
    "cA2": "#2B9CD7",
    "cD2": "#A977A6",
    "cD1": "#DB7268",
}
ABLATION_ALIGNED_DWT = {
    "muscle_ir": "dwt_global_stats_db2",
    "beta_cell": "dwt_energy_entropy_db2",
    "incretin_effect": "dwt_frequency_component_stats_db2",
    "hepatic_ir": "dwt_energy_entropy_db2",
}
METHOD_ORDER = ["DWT features", "Extracted features", "FFT features", "Welch PSD features"]
METHOD_COLORS = {
    "DWT features": "#456F9E",
    "Extracted features": "#A478B9",
    "FFT features": "#D04A5A",
    "Welch PSD features": "#F7B140",
}

def make_classifiers() -> dict[str, object]:
    return {
        "L1_logistic": LogisticRegression(C=10, penalty="l1", solver="saga", max_iter=10000, random_state=0),
        "L2_Logistic": LogisticRegression(C=10, penalty="l2", solver="saga", max_iter=10000, random_state=0),
        "RBF_SVC": SVC(kernel="rbf", C=10, probability=True, random_state=0),
        "RFR": RandomForestClassifier(random_state=23234),
    }


def run_repeated_stratified_cv(
    table: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    extraction: str,
    features_name: str,
    n_iterations: int = N_ITERATIONS,
    n_folds: int = N_FOLDS,
) -> pd.DataFrame:
    cfg = TARGETS[target]
    X = table[feature_cols].to_numpy(dtype=float)
    y_arr = table[cfg["label_col"]].to_numpy()
    subject_ids = table["subject_id"].astype(str).to_numpy()
    random.seed(RANDOM_SEED)
    classifiers = make_classifiers()
    tune_para = list(PARAMETERS_GRID.keys())
    groups = LabelEncoder().fit_transform(subject_ids)
    rows = []

    for iteration in range(n_iterations):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=iteration)
        group_u, ind_p = np.unique(groups, return_index=True)
        hyp_index = list(range(n_folds))
        random.shuffle(hyp_index)

        for fold, (train_index, validation_index) in enumerate(skf.split(X[ind_p], y_arr[ind_p])):
            val_idx = np.where(np.isin(groups, group_u[validation_index]))[0]
            train_idx = np.where(np.isin(groups, group_u[train_index]))[0]
            x_train = X[train_idx]
            x_val = X[val_idx]
            medians = np.nanmedian(x_train, axis=0)
            medians = np.where(np.isfinite(medians), medians, 0.0)
            x_train = np.where(np.isnan(x_train), medians, x_train)
            x_val = np.where(np.isnan(x_val), medians, x_val)
            y_train = y_arr[train_idx].ravel()
            y_val = y_arr[val_idx].ravel()

            for index, name in enumerate(CLASSIFIER_ORDER):
                classifier = classifiers[name]
                tun_parameter = tune_para[PARAIND_CV[index]]
                if tun_parameter == "C":
                    value_para = CSET_CV[name][hyp_index[fold]]
                else:
                    value_para = PARAMETERS_GRID[tun_parameter][hyp_index[fold]]
                classifier.set_params(**{tun_parameter: value_para})
                hyperparameter = f"{tun_parameter}={value_para}"

                label_encoder = LabelEncoder()
                classifier.fit(x_train, label_encoder.fit_transform(y_train))
                y_pred = label_encoder.inverse_transform(classifier.predict(x_val))
                y_proba = classifier.predict_proba(x_val)

                accuracy = metrics.accuracy_score(y_val, y_pred)
                recall_flag = len(np.unique(y_val)) == 1
                precision_flag = len(np.unique(y_pred)) == 1
                if recall_flag:
                    auc_val = float("nan")
                    recall = float("nan")
                elif y_proba.shape[1] == 2:
                    auc_val = metrics.roc_auc_score(y_val, y_proba[:, 1], multi_class="ovr")
                    recall = metrics.recall_score(y_val, y_pred, pos_label=cfg["pos_label"])
                else:
                    auc_val = metrics.roc_auc_score(y_val, y_proba, multi_class="ovr")
                    recall = metrics.recall_score(y_val, y_pred, average="micro")

                if precision_flag:
                    precision = float("nan")
                elif y_proba.shape[1] == 2:
                    precision = metrics.precision_score(y_val, y_pred, pos_label=cfg["pos_label"])
                else:
                    precision = metrics.precision_score(y_val, y_pred, average="micro")

                if recall_flag or precision_flag:
                    f1 = float("nan")
                elif y_proba.shape[1] == 2:
                    f1 = metrics.f1_score(y_val, y_pred, pos_label=cfg["pos_label"])
                else:
                    f1 = metrics.f1_score(y_val, y_pred, average="micro")

                rows.append(
                    {
                        "Features": features_name,
                        "Extraction": extraction,
                        "Iteration": iteration,
                        "Fold": fold,
                        "Classifier": name,
                        "hyperparameter": hyperparameter,
                        "auROC": round(auc_val, 4) if not np.isnan(auc_val) else float("nan"),
                        "Accuracy": round(accuracy, 4),
                        "Recall": round(recall, 4) if not np.isnan(recall) else float("nan"),
                        "Precision": round(precision, 4) if not np.isnan(precision) else float("nan"),
                        "F1": round(f1, 4) if not np.isnan(f1) else float("nan"),
                    }
                )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


# load data
CV_DIR.mkdir(parents=True, exist_ok=True)
handcrafted_features = pd.read_csv(FEATURE_DIR / "handcrafted_ogtt_features.csv")
dwt_full_features = pd.read_csv(FEATURE_DIR / "dwt_full_features_level2.csv")
dwt_frequency_ablation_features = pd.read_csv(FEATURE_DIR / "dwt_frequency_ablation_features_level2.csv")
fft_welch_features = pd.read_csv(FEATURE_DIR / "fft_welch_features.csv")

annotation = pd.read_csv(ANNOTATION_CSV, na_values=["NA", "--", ""])
initial_subjects = np.unique(annotation.loc[annotation["exp_type"].eq(INITIAL_EXP_TYPE), "subject_id"].astype(str))
initial_phenotypes = pd.read_csv(INITIAL_PHENOTYPES_CSV, na_values=["NA", "--", ""]).rename(
    columns={"SubjectID": "subject_id"}
)

target_label_tables = {}
for target, cfg in TARGETS.items():
    label_col = cfg["label_col"]
    if cfg["source"] == "annotation":
        labels = annotation[["subject_id", label_col]].drop_duplicates(subset=["subject_id"])
    else:
        labels = initial_phenotypes[["subject_id", label_col]].copy()
    labels["subject_id"] = labels["subject_id"].astype(str)
    target_label_tables[target] = labels.dropna(subset=[label_col]).reset_index(drop=True)


# repeated stratified CV for extracted, DWT, DWT ablation, FFT, and Welch features
results_by_target = {}
for target in TARGET_ORDER:
    cfg = TARGETS[target]
    labels = target_label_tables[target]
    result_frames = []
    target_dir = CV_DIR / target
    target_dir.mkdir(parents=True, exist_ok=True)

    merged = handcrafted_features.merge(labels, on="subject_id", how="inner")
    merged = merged[merged["subject_id"].isin(initial_subjects)].reset_index(drop=True)
    excluded_columns = METADATA_COLUMNS | {cfg["label_col"]}
    feature_cols = [
        col for col in merged.columns
        if col not in excluded_columns
        and pd.api.types.is_numeric_dtype(merged[col])
        and not merged[col].isna().all()
    ]
    result_frames.append(
        run_repeated_stratified_cv(
            merged,
            feature_cols,
            target,
            "handcrafted_ogtt_features",
            "venous_without_matching",
            N_ITERATIONS,
            N_FOLDS,
        )
    )

    for extraction in dwt_full_features["extraction"].drop_duplicates():
        one_feature_table = dwt_full_features[dwt_full_features["extraction"].eq(extraction)].copy()
        merged = one_feature_table.merge(labels, on="subject_id", how="inner")
        merged = merged[merged["subject_id"].isin(initial_subjects)].reset_index(drop=True)
        excluded_columns = METADATA_COLUMNS | {cfg["label_col"]}
        feature_cols = [
            col for col in merged.columns
            if col not in excluded_columns
            and pd.api.types.is_numeric_dtype(merged[col])
            and not merged[col].isna().all()
        ]
        result_frames.append(
            run_repeated_stratified_cv(
                merged,
                feature_cols,
                target,
                extraction,
                "venous_without_matching",
                N_ITERATIONS,
                N_FOLDS,
            )
        )

    for extraction in dwt_frequency_ablation_features["extraction"].drop_duplicates():
        one_feature_table = dwt_frequency_ablation_features[
            dwt_frequency_ablation_features["extraction"].eq(extraction)
        ].copy()
        merged = one_feature_table.merge(labels, on="subject_id", how="inner")
        merged = merged[merged["subject_id"].isin(initial_subjects)].reset_index(drop=True)
        excluded_columns = METADATA_COLUMNS | {cfg["label_col"]}
        feature_cols = [
            col for col in merged.columns
            if col not in excluded_columns
            and pd.api.types.is_numeric_dtype(merged[col])
            and not merged[col].isna().all()
        ]
        result_frames.append(
            run_repeated_stratified_cv(
                merged,
                feature_cols,
                target,
                extraction,
                "venous_without_matching",
                N_ITERATIONS,
                N_FOLDS,
            )
        )

    merged = fft_welch_features.merge(labels, on="subject_id", how="inner")
    merged = merged[merged["subject_id"].isin(initial_subjects)].reset_index(drop=True)
    result_frames.append(
        run_repeated_stratified_cv(
            merged,
            ["fft_max_amplitude", "fft_dominant_frequency", "fft75_frequency"],
            target,
            "fft_direct",
            "venous_without_matching",
            N_ITERATIONS,
            N_FOLDS,
        )
    )
    result_frames.append(
        run_repeated_stratified_cv(
            merged,
            ["psd_max_amplitude"],
            target,
            "welch_psd",
            "venous_without_matching",
            N_ITERATIONS,
            N_FOLDS,
        )
    )

    target_results = pd.concat(result_frames, ignore_index=True)
    target_results.to_csv(target_dir / "dwt_frequency_cv_results.csv", index=False)
    results_by_target[target] = target_results


# paired per-frequency contribution from DWT ablation results
PLOT_DIR.mkdir(parents=True, exist_ok=True)
frequency_contribution_rows = []
for target in TARGET_ORDER:
    result = results_by_target[target]
    full_extraction = ABLATION_ALIGNED_DWT[target]
    full_candidates = result[result["Extraction"].eq(full_extraction)].copy()
    full_candidates["auROC"] = pd.to_numeric(full_candidates["auROC"], errors="coerce")
    best_summary = (
        full_candidates.groupby("Classifier")["auROC"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["mean", "std"], ascending=[False, True])
    )
    best_classifier = best_summary.iloc[0]["Classifier"]

    full = result[
        result["Extraction"].eq(full_extraction)
        & result["Classifier"].eq(best_classifier)
    ][["Iteration", "Fold", "Classifier", "hyperparameter", "auROC"]].rename(columns={"auROC": "full_auROC"})

    for component in FREQUENCY_COMPONENT_ORDER:
        removed_extraction = f"{full_extraction}_without_{component}"
        removed = result[
            result["Extraction"].eq(removed_extraction)
            & result["Classifier"].eq(best_classifier)
        ][["Iteration", "Fold", "Classifier", "hyperparameter", "auROC"]].rename(columns={"auROC": "removed_auROC"})
        paired = full.merge(removed, on=["Iteration", "Fold", "Classifier", "hyperparameter"], how="inner")
        delta = paired["full_auROC"].astype(float) - paired["removed_auROC"].astype(float)
        frequency_contribution_rows.append(
            {
                "target": target,
                "extraction": full_extraction,
                "classifier": best_classifier,
                "removed_frequency_component": component,
                "contribution_auroc": float(delta.mean()),
                "sd_delta_auROC": float(delta.std(ddof=1)),
                "n_pairs": int(delta.count()),
            }
        )

frequency_contribution = pd.DataFrame(frequency_contribution_rows)
frequency_contribution["ci95_delta_auROC"] = (
    1.96 * frequency_contribution["sd_delta_auROC"] / np.sqrt(frequency_contribution["n_pairs"])
)

# plot figure 2a
feature_group_rows = []
for target in TARGET_ORDER:
    result = results_by_target[target]
    method_extractions = {
        "DWT features": ABLATION_ALIGNED_DWT[target],
        "Extracted features": "handcrafted_ogtt_features",
        "FFT features": "fft_direct",
        "Welch PSD features": "welch_psd",
    }
    for method in METHOD_ORDER:
        extraction = method_extractions[method]
        one_method = result[result["Extraction"].eq(extraction)].copy()
        one_method["auROC"] = pd.to_numeric(one_method["auROC"], errors="coerce")
        method_summary = (
            one_method.groupby("Classifier")["auROC"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values(["mean", "std"], ascending=[False, True])
        )
        best = method_summary.iloc[0]
        feature_group_rows.append(
            {
                "target": target,
                "method": method,
                "extraction": extraction,
                "classifier": best["Classifier"],
                "mean_auROC": float(best["mean"]),
                "sd_auROC": float(best["std"]),
                "n": int(best["count"]),
                "ci95_auROC": 1.96 * float(best["std"]) / np.sqrt(int(best["count"])),
            }
        )

feature_group_auroc_summary = pd.DataFrame(feature_group_rows)

fig, ax = plt.subplots(figsize=(10.4, 5.6))
x = np.arange(len(TARGET_ORDER))
width = 0.18
offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHOD_ORDER))
for offset, method in zip(offsets, METHOD_ORDER):
    one_method = feature_group_auroc_summary[
        feature_group_auroc_summary["method"].eq(method)
    ].set_index("target").loc[TARGET_ORDER].reset_index()
    ax.bar(
        x + offset,
        one_method["mean_auROC"].to_numpy(dtype=float),
        width=width,
        yerr=one_method["ci95_auROC"].to_numpy(dtype=float),
        capsize=3,
        label=method,
        color=METHOD_COLORS[method],
        edgecolor="#0F172A",
        linewidth=0.6,
        error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#334155"},
    )
ax.set_xticks(x)
ax.set_xticklabels([TARGET_LABELS[target] for target in TARGET_ORDER], fontsize=10)
ax.set_ylabel("Mean AUROC", fontsize=11)
ax.set_ylim(0.0, 1.05)
ax.axhline(0.5, color="#64748B", linestyle="--", linewidth=1.0, alpha=0.9)
ax.text(len(TARGET_ORDER) - 0.18, 0.515, "AUROC = 0.5", color="#475569", fontsize=8, ha="right", va="bottom")
ax.set_title("Glycemic Endotypes Classification Results", fontsize=13, pad=12)
ax.grid(axis="y", alpha=0.22, linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(PLOT_DIR / "feature_group_auroc_barplot_ablation_aligned_ci95_auroc05.png", dpi=300)
plt.close(fig)

print(f"CV fold-level CSVs and plots saved to: {CV_DIR}")

# plot figure 2b
fig, ax = plt.subplots(figsize=(8.0, 3.8))
x = np.arange(len(TARGET_ORDER))
width = 0.22
offsets = np.linspace(-width, width, len(FREQUENCY_COMPONENT_ORDER))
for offset, component in zip(offsets, FREQUENCY_COMPONENT_ORDER):
    one_component = frequency_contribution[
        frequency_contribution["removed_frequency_component"].eq(component)
    ].set_index("target")
    values = [one_component.loc[target, "contribution_auroc"] for target in TARGET_ORDER]
    errors = [one_component.loc[target, "ci95_delta_auROC"] for target in TARGET_ORDER]
    ax.bar(
        x + offset,
        values,
        width=width,
        yerr=errors,
        capsize=3,
        label=FREQUENCY_COMPONENT_LABELS[component],
        color=FREQUENCY_COMPONENT_COLORS[component],
        edgecolor="#0F172A",
        linewidth=0.6,
        error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#334155"},
    )
ax.axhline(0, color="#2F3437", linewidth=0.8)
ax.set_title("DWT Frequency Component Contributions", fontsize=13, pad=12)
ax.set_ylabel("AUROC loss after component removal", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([TARGET_LABELS[target].replace("\n", " ") for target in TARGET_ORDER], fontsize=10)
lower = min(0.0, (frequency_contribution["contribution_auroc"] - frequency_contribution["ci95_delta_auROC"]).min() * 1.10)
upper = max((frequency_contribution["contribution_auroc"] + frequency_contribution["ci95_delta_auROC"]).max() * 1.10, 0.05)
ax.set_ylim(lower, upper)
ax.grid(axis="y", alpha=0.22, linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(PLOT_DIR / "wavelet_frequency_contribution_barplot_ci95.png", dpi=300, bbox_inches="tight")
plt.close(fig)