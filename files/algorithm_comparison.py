from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVR

from goal_difference_model import build_model_data as build_model_2_1_data
from goals_scored_model import build_model_data as build_model_2_2_data


PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def selected_features(model_number):
    if model_number == "2_1":
        data = pd.read_csv(PROCESSED_DIR / "model_2_1_final_104.csv")
        return [
            column
            for column in data.columns
            if column not in {"Match_ID", "Goal_Difference"}
        ]

    data = pd.read_csv(PROCESSED_DIR / "model_2_2_final_208.csv")
    return [
        column
        for column in data.columns
        if column not in {"Observation_ID", "Goals_Scored"}
    ]


def selected_linear_pipeline(model_number):
    metrics_file = (
        RESULTS_DIR
        / f"model_{model_number}_final_metrics.csv"
    )
    preprocessing = pd.read_csv(metrics_file).iloc[0]["Preprocessing"]

    if preprocessing == "OLS":
        return "Linear Regression", Pipeline(
            [("model", LinearRegression())]
        )

    if preprocessing == "Standardized OLS":
        return "Linear Regression (standardized)", Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )

    if preprocessing == "Yeo-Johnson OLS":
        return "Linear Regression (Yeo-Johnson)", Pipeline(
            [
                (
                    "transformer",
                    PowerTransformer(
                        method="yeo-johnson",
                        standardize=True,
                    ),
                ),
                ("model", LinearRegression()),
            ]
        )

    raise ValueError(f"Unknown preprocessing: {preprocessing}")


def build_algorithms(model_number):
    linear_name, linear_model = selected_linear_pipeline(model_number)

    return {
        linear_name: linear_model,
        "Random Forest": Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        random_state=42,
                    ),
                )
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                (
                    "model",
                    GradientBoostingRegressor(
                        random_state=42,
                    ),
                )
            ]
        ),
        "Support Vector Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf")),
            ]
        ),
    }


def summarise_cv(scores):
    return {
        "CV_MAE": (-scores["test_MAE"]).mean(),
        "CV_RMSE": np.sqrt(-scores["test_MSE"]).mean(),
        "CV_R2": scores["test_R2"].mean(),
    }


def holdout_metrics(y_true, predictions):
    mse = mean_squared_error(y_true, predictions)
    return {
        "Holdout_MAE": mean_absolute_error(y_true, predictions),
        "Holdout_RMSE": np.sqrt(mse),
        "Holdout_R2": r2_score(y_true, predictions),
    }


def save_comparison_figure(results, title, output_path):
    plot_data = results.sort_values("CV_RMSE").reset_index(drop=True)
    x = np.arange(len(plot_data))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(
        x - width / 2,
        plot_data["CV_RMSE"],
        width,
        label="Cross-validation RMSE",
    )
    plt.bar(
        x + width / 2,
        plot_data["Holdout_RMSE"],
        width,
        label="Holdout RMSE",
    )
    plt.xticks(
        x,
        plot_data["Algorithm"],
        rotation=15,
        ha="right",
    )
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def compare_model_2_1():
    data = build_model_2_1_data()
    features = selected_features("2_1")

    train_index, test_index = train_test_split(
        np.arange(len(data)),
        test_size=0.20,
        random_state=42,
    )
    train_data = data.iloc[train_index].copy()
    test_data = data.iloc[test_index].copy()

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for name, algorithm in build_algorithms("2_1").items():
        scores = cross_validate(
            algorithm,
            train_data[features],
            train_data["Goal_Difference"],
            cv=cv,
            scoring={
                "MAE": "neg_mean_absolute_error",
                "MSE": "neg_mean_squared_error",
                "R2": "r2",
            },
        )

        algorithm.fit(
            train_data[features],
            train_data["Goal_Difference"],
        )
        predictions = algorithm.predict(test_data[features])

        rows.append(
            {
                "Algorithm": name,
                **summarise_cv(scores),
                **holdout_metrics(
                    test_data["Goal_Difference"],
                    predictions,
                ),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["CV_RMSE", "CV_MAE"],
        ascending=True,
    ).reset_index(drop=True)
    results["CV_Rank"] = np.arange(1, len(results) + 1)
    results.to_csv(
        RESULTS_DIR / "model_2_1_algorithm_comparison.csv",
        index=False,
    )

    save_comparison_figure(
        results,
        "Model 2.1: Regression Algorithm Comparison",
        FIGURES_DIR / "model_2_1_algorithm_comparison.png",
    )

    return results


def compare_model_2_2():
    data = build_model_2_2_data()
    features = selected_features("2_2")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )
    train_index, test_index = next(
        splitter.split(
            data,
            data["Goals_Scored"],
            groups=data["Match_ID"],
        )
    )

    train_data = data.iloc[train_index].copy()
    test_data = data.iloc[test_index].copy()
    cv = GroupKFold(n_splits=5)
    rows = []

    for name, algorithm in build_algorithms("2_2").items():
        scores = cross_validate(
            algorithm,
            train_data[features],
            train_data["Goals_Scored"],
            groups=train_data["Match_ID"],
            cv=cv,
            scoring={
                "MAE": "neg_mean_absolute_error",
                "MSE": "neg_mean_squared_error",
                "R2": "r2",
            },
        )

        algorithm.fit(
            train_data[features],
            train_data["Goals_Scored"],
        )
        predictions = algorithm.predict(test_data[features])

        rows.append(
            {
                "Algorithm": name,
                **summarise_cv(scores),
                **holdout_metrics(
                    test_data["Goals_Scored"],
                    predictions,
                ),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["CV_RMSE", "CV_MAE"],
        ascending=True,
    ).reset_index(drop=True)
    results["CV_Rank"] = np.arange(1, len(results) + 1)
    results.to_csv(
        RESULTS_DIR / "model_2_2_algorithm_comparison.csv",
        index=False,
    )

    save_comparison_figure(
        results,
        "Model 2.2: Regression Algorithm Comparison",
        FIGURES_DIR / "model_2_2_algorithm_comparison.png",
    )

    return results


def main():
    model_2_1_results = compare_model_2_1()
    model_2_2_results = compare_model_2_2()

    print("Algorithm comparison complete")
    print("\nModel 2.1 comparison:")
    print(model_2_1_results.round(4).to_string(index=False))
    print("\nModel 2.2 comparison:")
    print(model_2_2_results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
