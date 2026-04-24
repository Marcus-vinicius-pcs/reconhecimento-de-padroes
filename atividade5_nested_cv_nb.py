from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCORING = {
    "accuracy": make_scorer(accuracy_score),
    "f1": make_scorer(f1_score),
    "recall": make_scorer(recall_score),
    "precision": make_scorer(precision_score),
}


@dataclass
class ModelSpec:
    key: str
    label: str
    estimator: Any
    param_grid: dict[str, list[Any]]


class AdultFeatureBuilder(BaseEstimator, TransformerMixin):
    """Feature engineering that is fit only on training folds."""

    education_map = {
        "Preschool": "dropout",
        "1st-4th": "dropout",
        "5th-6th": "dropout",
        "7th-8th": "dropout",
        "9th": "dropout",
        "10th": "dropout",
        "11th": "dropout",
        "12th": "dropout",
        "HS-grad": "HighGrad",
        "Some-college": "CommunityCollege",
        "Assoc-acdm": "CommunityCollege",
        "Assoc-voc": "CommunityCollege",
        "Bachelors": "Bachelors",
        "Masters": "Masters",
        "Prof-school": "Masters",
        "Doctorate": "Doctorate",
    }

    marital_map = {
        "Never-married": "NotMarried",
        "Married-civ-spouse": "Married",
        "Married-AF-spouse": "Married",
        "Married-spouse-absent": "NotMarried",
        "Divorced": "Separated",
        "Separated": "Separated",
        "Widowed": "Widowed",
    }

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "AdultFeatureBuilder":
        frame = self._prepare_frame(X)
        if "native-country" in frame.columns:
            counts = frame["native-country"].dropna().value_counts()
            self.top_countries_ = counts.head(3).index.tolist()
        else:
            self.top_countries_ = []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._prepare_frame(X)

        if "education" in frame.columns:
            frame["education"] = frame["education"].replace(self.education_map)

        if "marital-status" in frame.columns:
            frame["marital-status"] = frame["marital-status"].replace(self.marital_map)

        for col in ["capital-gain", "capital-loss"]:
            if col in frame.columns:
                frame[col] = np.log1p(frame[col])

        if {"capital-gain", "capital-loss"}.issubset(frame.columns):
            frame["capital-balance"] = frame["capital-gain"] - frame["capital-loss"]
            frame = frame.drop(columns=["capital-gain", "capital-loss"])

        if "native-country" in frame.columns and self.top_countries_:
            frame["native-country"] = frame["native-country"].where(
                frame["native-country"].isin(self.top_countries_),
                "Other",
            )

        drop_cols = [col for col in ["fnlwgt", "educational-num"] if col in frame.columns]
        if drop_cols:
            frame = frame.drop(columns=drop_cols)

        return frame

    @staticmethod
    def _prepare_frame(X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        object_cols = frame.select_dtypes(include=["object", "string"]).columns.tolist()
        for col in object_cols:
            frame[col] = frame[col].astype(str).str.strip()
            frame.loc[frame[col] == "?", col] = np.nan
        return frame


def get_dataset_cache_path() -> Path:
    return (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "datasets"
        / "wenruliu"
        / "adult-income-dataset"
        / "versions"
        / "2"
        / "adult.csv"
    )


def load_adult_dataframe() -> pd.DataFrame:
    dataset_path = get_dataset_cache_path()
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Adult dataset not found in Kaggle cache. Download it first with kagglehub."
        )
    return pd.read_csv(dataset_path, compression="zip", encoding="utf-8")


def clean_rows_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = df.copy()
    frame["income"] = frame["income"].astype(str).str.replace(".", "", regex=False).str.strip()

    object_cols = frame.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in object_cols:
        frame[col] = frame[col].astype(str).str.strip()
        frame.loc[frame[col] == "?", col] = np.nan

    frame = frame.drop_duplicates().reset_index(drop=True)

    feature_cols = [col for col in frame.columns if col != "income"]
    label_noise = frame.groupby(feature_cols, dropna=False)["income"].transform("nunique") > 1
    frame = frame.loc[~label_noise].reset_index(drop=True)

    income_map = {"<=50K": 0, ">50K": 1}
    frame["income"] = frame["income"].map(income_map)
    frame = frame.dropna(subset=["income"]).reset_index(drop=True)
    frame["income"] = frame["income"].astype(int)

    X = frame.drop(columns=["income"])
    y = frame["income"]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_selector = make_column_selector(dtype_include=np.number)
    categorical_selector = make_column_selector(dtype_include=["object", "string"])

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_selector),
            ("categorical", categorical_pipeline, categorical_selector),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def build_model_specs() -> list[ModelSpec]:
    common_steps = [
        ("features", AdultFeatureBuilder()),
        ("preprocessor", build_preprocessor()),
        ("scaler", StandardScaler()),
    ]

    full_model = Pipeline(
        steps=[
            *common_steps,
            ("nb", GaussianNB()),
        ]
    )

    pca_model = Pipeline(
        steps=[
            *common_steps,
            ("pca", PCA()),
            ("nb", GaussianNB()),
        ]
    )

    select_model = Pipeline(
        steps=[
            *common_steps,
            ("selector", SelectKBest(score_func=f_classif)),
            ("nb", GaussianNB()),
        ]
    )

    select_under_model = ImbPipeline(
        steps=[
            *common_steps,
            ("selector", SelectKBest(score_func=f_classif)),
            ("undersampler", RandomUnderSampler(random_state=42)),
            ("nb", GaussianNB()),
        ]
    )

    return [
        ModelSpec(
            key="all_features",
            label="All features",
            estimator=full_model,
            param_grid={
                "nb__var_smoothing": [1e-9, 1e-8],
            },
        ),
        ModelSpec(
            key="pca",
            label="PCA",
            estimator=pca_model,
            param_grid={
                "pca__n_components": [10, 20],
                "nb__var_smoothing": [1e-9, 1e-8],
            },
        ),
        ModelSpec(
            key="select_k_best",
            label="SelectKBest",
            estimator=select_model,
            param_grid={
                "selector__k": [10, "all"],
                "nb__var_smoothing": [1e-9, 1e-8],
            },
        ),
        ModelSpec(
            key="select_k_best_under",
            label="SelectKBest + RandomUnderSampler",
            estimator=select_under_model,
            param_grid={
                "selector__k": [10, "all"],
                "nb__var_smoothing": [1e-9, 1e-8],
            },
        ),
    ]


def metric_summary(scores: np.ndarray) -> dict[str, float]:
    arr = np.asarray(scores, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
    }


def evaluate_model(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    outer_splits: int,
    inner_splits: int,
    random_state: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inner_cv = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=random_state,
    )
    outer_cv = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=random_state,
    )

    search = GridSearchCV(
        estimator=spec.estimator,
        param_grid=spec.param_grid,
        cv=inner_cv,
        scoring="f1",
        refit=True,
        n_jobs=1,
        error_score="raise",
    )

    nested = cross_validate(
        estimator=search,
        X=X,
        y=y,
        cv=outer_cv,
        scoring=SCORING,
        return_estimator=True,
        n_jobs=1,
        error_score="raise",
    )

    final_search = GridSearchCV(
        estimator=spec.estimator,
        param_grid=spec.param_grid,
        cv=inner_cv,
        scoring="f1",
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    final_search.fit(X, y)

    summary: dict[str, Any] = {
        "model_key": spec.key,
        "model_label": spec.label,
        "classifier": "GaussianNB",
        "selected_params_full_data": final_search.best_params_,
        "selected_f1_full_data": float(final_search.best_score_),
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
    }

    fold_rows: list[dict[str, Any]] = []
    for fold_idx in range(outer_splits):
        row = {
            "model_key": spec.key,
            "model_label": spec.label,
            "fold": fold_idx + 1,
            "best_params": nested["estimator"][fold_idx].best_params_,
        }
        for metric in SCORING:
            score = float(nested[f"test_{metric}"][fold_idx])
            row[metric] = score
        fold_rows.append(row)

    for metric in SCORING:
        stats = metric_summary(nested[f"test_{metric}"])
        summary[f"{metric}_mean"] = stats["mean"]
        summary[f"{metric}_std"] = stats["std"]
        summary[f"{metric}_ci95"] = stats["ci95"]

    summary["outer_best_params"] = [row["best_params"] for row in fold_rows]
    return summary, fold_rows


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_result_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary_df.iterrows():
        rows.append(
            {
                "Dataset variant": row["model_label"],
                "Classifier": row["classifier"],
                "Best params on full data": json.dumps(row["selected_params_full_data"]),
                "F1": format_metric(row, "f1"),
                "Accuracy": format_metric(row, "accuracy"),
                "Recall": format_metric(row, "recall"),
                "Precision": format_metric(row, "precision"),
            }
        )
    return pd.DataFrame(rows)


def format_metric(row: pd.Series, metric: str) -> str:
    return (
        f"{row[f'{metric}_mean']:.4f} "
        f"(sd={row[f'{metric}_std']:.4f}; ci95={row[f'{metric}_ci95']:.4f})"
    )


def generate_bar_plots(summary_df: pd.DataFrame, output_dir: Path) -> list[str]:
    generated_files: list[str] = []
    x_labels = summary_df["model_label"].tolist()
    x_pos = np.arange(len(x_labels))

    for metric in SCORING:
        for error_kind in ["std", "ci95"]:
            fig, ax = plt.subplots(figsize=(11, 6))
            heights = summary_df[f"{metric}_mean"].to_numpy(dtype=float)
            errors = summary_df[f"{metric}_{error_kind}"].to_numpy(dtype=float)

            bars = ax.bar(
                x_pos,
                heights,
                yerr=errors,
                capsize=8,
                color="#4c78a8",
                edgecolor="black",
                label="GaussianNB",
            )

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=15, ha="right")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(
                f"GaussianNB - {metric.capitalize()} by dataset variant ({error_kind})"
            )
            ax.set_ylim(0.0, min(1.05, max(heights + errors) + 0.08))
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.3)

            for bar, height in zip(bars, heights):
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            file_name = f"bar_{metric}_{error_kind}.png"
            file_path = output_dir / file_name
            fig.tight_layout()
            fig.savefig(file_path, dpi=200)
            plt.close(fig)
            generated_files.append(file_name)

    return generated_files


def generate_markdown(
    summary_df: pd.DataFrame,
    plot_files: list[str],
    markdown_path: Path,
    outer_splits: int,
    inner_splits: int,
) -> None:
    table_df = build_result_table(summary_df)
    markdown_table = dataframe_to_markdown(table_df)

    lines = [
        "# Atividade 5 - Results",
        "",
        "## Result 1 - Table",
        "",
        "Each row below shows the nested stratified cross-validation estimate for GaussianNB.",
        "The values use mean, standard deviation, and 95% confidence interval from the outer folds.",
        "",
        markdown_table,
        "",
        "## Result 2 - Bar charts",
        "",
        "The project now has one chart for each performance metric.",
        "For each metric there are two versions of the chart:",
        "- `std`: error bars with standard deviation across outer folds.",
        "- `ci95`: error bars with 95% confidence interval across outer folds.",
        "",
        "Use the `ci95` files if the presentation asks for confidence intervals.",
        "Use the `std` files if the presentation asks for standard deviation.",
        "",
        "Suggested legend text for the slide:",
        "",
        "> Bars show the mean performance of GaussianNB for each dataset variant. Error bars show either the standard deviation or the 95% confidence interval across the outer folds of nested stratified cross-validation.",
        "",
        "Generated chart files:",
    ]

    for plot_file in plot_files:
        lines.append(f"- `{plot_file}`")

    lines.extend(
        [
            "",
            "Suggested labels for the four dataset variants:",
            "- `All features`: feature engineering + imputation + encoding + scaling, without dimensionality reduction.",
            "- `PCA`: same preprocessing plus PCA inside the training folds.",
            "- `SelectKBest`: same preprocessing plus supervised feature selection inside the training folds.",
            "- `SelectKBest + RandomUnderSampler`: same preprocessing plus feature selection and class balancing inside the training folds.",
            "",
            "Method note for the presentation:",
            f"- Performance estimation used nested stratified cross-validation with outer={outer_splits} folds and inner={inner_splits} folds.",
            "- Hyperparameter tuning used the inner cross-validation loop with F1-score as the optimization criterion.",
            "- Performance reporting used the outer cross-validation loop with F1, accuracy, recall, and precision.",
            "- Imputation, encoding, scaling, feature selection, PCA, and undersampling were all fit only on the training folds to avoid data leakage.",
            "- Because of hardware limits, the final experiment used a reduced search space and fewer folds than the ideal 10x10 configuration.",
        ]
    )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = df.columns.tolist()
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = []
    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in headers]
        body_rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header_row, separator_row, *body_rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-splits", type=int, default=10)
    parser.add_argument("--inner-splits", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("atividade5_outputs"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_adult_dataframe()
    X, y = clean_rows_and_target(df)

    summaries: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for spec in build_model_specs():
        summary, folds = evaluate_model(
            spec=spec,
            X=X,
            y=y,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            random_state=args.random_state,
        )
        summaries.append(summary)
        fold_rows.extend(folds)

    summary_df = pd.DataFrame(summaries).sort_values("model_label").reset_index(drop=True)
    folds_df = pd.DataFrame(fold_rows).sort_values(["model_label", "fold"]).reset_index(drop=True)

    summary_df.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    folds_df.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    save_json(args.output_dir / "summary_metrics.json", summaries)
    save_json(args.output_dir / "fold_metrics.json", fold_rows)

    plot_files = generate_bar_plots(summary_df, args.output_dir)
    generate_markdown(
        summary_df=summary_df,
        plot_files=plot_files,
        markdown_path=args.output_dir / "atividade5_apresentacao.md",
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
    )

    print("Saved outputs to:", args.output_dir.resolve())
    print(summary_df[["model_label", "f1_mean", "accuracy_mean", "recall_mean", "precision_mean"]])


if __name__ == "__main__":
    main()
