from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler


PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


CANDIDATE_FEATURES = [
    "FIFA_Points_Diff",
    "FIFA_Rank_Diff",
    "Prior_GF_Avg_Diff",
    "Prior_GA_Avg_Diff",
    "Prior_GD_Avg_Diff",
    "Prior_Win_Rate_Diff",
    "Prior_Draw_Rate_Diff",
    "Prior_Points_Per_Game_Diff",
    "Prior_CleanSheet_Rate_Diff",
    "Prior_FailedScore_Rate_Diff",
    "Prior_Opp_FIFA_Points_Avg_Diff",
    "Prior_Strength_Adjusted_GD_Avg_Diff",
    "Rest_Days_Diff",
]


FEATURE_SETS = {
    "Initial model": [
        "FIFA_Points_Diff",
        "Prior_GF_Avg_Diff",
        "Prior_GA_Avg_Diff",
        "Prior_GD_Avg_Diff",
        "Prior_Win_Rate_Diff",
        "Prior_Draw_Rate_Diff",
        "Prior_CleanSheet_Rate_Diff",
        "Rest_Days_Diff",
    ],
    "Balanced form": [
        "FIFA_Points_Diff",
        "Prior_GF_Avg_Diff",
        "Prior_GA_Avg_Diff",
        "Prior_Win_Rate_Diff",
        "Prior_Draw_Rate_Diff",
        "Prior_CleanSheet_Rate_Diff",
        "Prior_FailedScore_Rate_Diff",
        "Rest_Days_Diff",
    ],
    "Opponent-strength adjusted": [
        "FIFA_Points_Diff",
        "Prior_GF_Avg_Diff",
        "Prior_GA_Avg_Diff",
        "Prior_Draw_Rate_Diff",
        "Prior_CleanSheet_Rate_Diff",
        "Prior_FailedScore_Rate_Diff",
        "Prior_Opp_FIFA_Points_Avg_Diff",
        "Prior_Strength_Adjusted_GD_Avg_Diff",
    ],
    "Goal difference form": [
        "FIFA_Points_Diff",
        "Prior_GD_Avg_Diff",
        "Prior_Win_Rate_Diff",
        "Prior_Draw_Rate_Diff",
        "Prior_Points_Per_Game_Diff",
        "Prior_CleanSheet_Rate_Diff",
        "Prior_FailedScore_Rate_Diff",
        "Rest_Days_Diff",
    ],
    "Points and rates": [
        "FIFA_Points_Diff",
        "Prior_GF_Avg_Diff",
        "Prior_GA_Avg_Diff",
        "Prior_Points_Per_Game_Diff",
        "Prior_Draw_Rate_Diff",
        "Prior_CleanSheet_Rate_Diff",
        "Prior_FailedScore_Rate_Diff",
        "Rest_Days_Diff",
    ],
}



def build_model_data():
    matches = pd.read_csv(PROCESSED_DIR / "matches_master_104.csv")
    team_data = pd.read_csv(PROCESSED_DIR / "prematch_team_features_208.csv")

    team_features = team_data[
        ["Match_ID", "Team", *CANDIDATE_FEATURES]
    ].copy()

    model_data = matches[
        [
            "Match_ID",
            "Date",
            "Round",
            "Team_A",
            "Team_B",
            "Goal_Difference",
        ]
    ].merge(
        team_features,
        left_on=["Match_ID", "Team_A"],
        right_on=["Match_ID", "Team"],
        how="left",
        validate="one_to_one",
    )

    model_data = model_data.drop(columns=["Team"])

    if len(model_data) != 104:
        raise ValueError("Model 2.1 must contain exactly 104 matches.")

    if model_data[CANDIDATE_FEATURES].isna().any().any():
        raise ValueError("Model 2.1 candidate predictors contain missing values.")

    return model_data


def save_eda(train_data):
    rows = []

    for feature in CANDIDATE_FEATURES:
        values = train_data[feature]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        rows.append(
            {
                "Feature": feature,
                "Available_N": values.notna().sum(),
                "Missing_N": values.isna().sum(),
                "Mean": values.mean(),
                "Median": values.median(),
                "Std_Dev": values.std(),
                "IQR_Outliers": (
                    (values < q1 - 1.5 * iqr)
                    | (values > q3 + 1.5 * iqr)
                ).sum(),
                "Correlation_With_Goal_Difference": values.corr(
                    train_data["Goal_Difference"]
                ),
            }
        )

    summary = pd.DataFrame(rows)
    summary["Absolute_Correlation"] = summary[
        "Correlation_With_Goal_Difference"
    ].abs()
    summary = summary.sort_values("Absolute_Correlation", ascending=False)
    summary.to_csv(
        RESULTS_DIR / "model_2_1_eda_summary.csv",
        index=False,
    )

    correlation = train_data[CANDIDATE_FEATURES].corr()
    correlation.to_csv(
        RESULTS_DIR / "model_2_1_predictor_correlations.csv"
    )

    plt.figure(figsize=(7, 5))
    plt.hist(train_data["Goal_Difference"], bins=10)
    plt.xlabel("Goal difference")
    plt.ylabel("Number of training matches")
    plt.title("Model 2.1 Training Goal-Difference Distribution")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "model_2_1_target_histogram.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(13, 9))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
    )
    plt.title("Model 2.1 Training Predictor Correlation Matrix")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "model_2_1_correlation_heatmap.png",
        dpi=180,
    )
    plt.close()


def save_linearity_plots(train_data, features):
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for axis, feature in zip(axes, features):
        axis.scatter(train_data[feature], train_data["Goal_Difference"])
        axis.set_xlabel(feature)
        axis.set_ylabel("Goal Difference")

    figure.suptitle("Model 2.1: Final Predictors vs Goal Difference (Training Data)")
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "model_2_1_linearity_scatterplots.png",
        dpi=180,
    )
    plt.close(figure)


def make_pipeline(variant):
    if variant == "OLS":
        return Pipeline([("model", LinearRegression())])

    if variant == "Standardized OLS":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )

    if variant == "Yeo-Johnson OLS":
        return Pipeline(
            [
                (
                    "transformer",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
                ("model", LinearRegression()),
            ]
        )

    raise ValueError("Unknown model variant.")


def cross_validation_results(train_data, features, variant):
    model = make_pipeline(variant)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        model,
        train_data[features],
        train_data["Goal_Difference"],
        cv=cv,
        scoring={
            "MAE": "neg_mean_absolute_error",
            "MSE": "neg_mean_squared_error",
            "R2": "r2",
        },
    )

    return {
        "CV_MAE": (-scores["test_MAE"]).mean(),
        "CV_RMSE": np.sqrt(-scores["test_MSE"]).mean(),
        "CV_R2": scores["test_R2"].mean(),
    }


def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def mean_baseline_results(train_data, test_data, features):
    baseline = DummyRegressor(strategy="mean")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        baseline,
        train_data[features],
        train_data["Goal_Difference"],
        cv=cv,
        scoring={
            "MAE": "neg_mean_absolute_error",
            "MSE": "neg_mean_squared_error",
            "R2": "r2",
        },
    )

    baseline.fit(
        train_data[features],
        train_data["Goal_Difference"],
    )
    predictions = baseline.predict(test_data[features])
    mse = mean_squared_error(
        test_data["Goal_Difference"],
        predictions,
    )

    return {
        "CV_MAE": (-scores["test_MAE"]).mean(),
        "CV_RMSE": np.sqrt(-scores["test_MSE"]).mean(),
        "CV_R2": scores["test_R2"].mean(),
        "Holdout_MAE": mean_absolute_error(
            test_data["Goal_Difference"],
            predictions,
        ),
        "Holdout_RMSE": np.sqrt(mse),
        "Holdout_R2": r2_score(
            test_data["Goal_Difference"],
            predictions,
        ),
    }


def save_coefficients_and_equation(final_model, selected_features, selected_variant):
    regression = final_model.named_steps["model"]

    coefficients = pd.DataFrame(
        {
            "Feature": selected_features,
            "Coefficient": regression.coef_,
        }
    )
    coefficients.to_csv(
        RESULTS_DIR / "model_2_1_coefficients.csv",
        index=False,
    )

    if selected_variant == "Yeo-Johnson OLS":
        equation_features = [
            f"YJ-standardized({feature})"
            for feature in selected_features
        ]
    elif selected_variant == "Standardized OLS":
        equation_features = [
            f"Z({feature})"
            for feature in selected_features
        ]
    else:
        equation_features = selected_features

    equation_parts = [f"{regression.intercept_:.4f}"]
    for feature, coefficient in zip(equation_features, regression.coef_):
        sign = "+" if coefficient >= 0 else "-"
        equation_parts.append(
            f" {sign} {abs(coefficient):.4f}({feature})"
        )

    equation = "Predicted Goal Difference = " + "".join(equation_parts)
    (RESULTS_DIR / "model_2_1_regression_equation.txt").write_text(
        equation,
        encoding="utf-8",
    )

    print("\nFinal regression equation:")
    print(equation)

    if selected_variant != "OLS":
        print("Note: coefficients are based on transformed and standardized predictors.")


def main():
    model_data = build_model_data()

    train_index, test_index = train_test_split(
        np.arange(len(model_data)),
        test_size=0.20,
        random_state=42,
    )

    train_data = model_data.iloc[train_index].copy()
    test_data = model_data.iloc[test_index].copy()

    save_eda(train_data)

    experiment_rows = []

    for name, features in FEATURE_SETS.items():
        scores = cross_validation_results(train_data, features, "OLS")
        correlation = train_data[features].corr()
        upper = correlation.where(
            np.triu(np.ones(correlation.shape), k=1).astype(bool)
        )

        experiment_rows.append(
            {
                "Model": name,
                "CV_MAE": scores["CV_MAE"],
                "CV_RMSE": scores["CV_RMSE"],
                "CV_R2": scores["CV_R2"],
                "Max_Absolute_Predictor_Correlation": upper.abs().max().max(),
                "Features": " | ".join(features),
            }
        )

    experiments = pd.DataFrame(experiment_rows)
    experiments["Eligible_After_Multicollinearity_Check"] = (
        experiments["Max_Absolute_Predictor_Correlation"] < 0.90
    )
    experiments = experiments.sort_values(
        ["CV_RMSE", "CV_MAE"],
        ascending=True,
    ).reset_index(drop=True)

    experiments.to_csv(
        RESULTS_DIR / "model_2_1_model_experiments.csv",
        index=False,
    )

    eligible = experiments[
        experiments["Max_Absolute_Predictor_Correlation"] < 0.90
    ].copy()

    if eligible.empty:
        raise ValueError(
            "No candidate feature set passed the multicollinearity check."
        )

    selected_name = eligible.iloc[0]["Model"]
    selected_features = FEATURE_SETS[selected_name]

    preprocessing_rows = []
    for variant in ["OLS", "Standardized OLS", "Yeo-Johnson OLS"]:
        scores = cross_validation_results(
            train_data,
            selected_features,
            variant,
        )
        preprocessing_rows.append({"Variant": variant, **scores})

    preprocessing = pd.DataFrame(preprocessing_rows).sort_values(
        ["CV_RMSE", "CV_MAE"],
        ascending=True,
    ).reset_index(drop=True)

    preprocessing.to_csv(
        RESULTS_DIR / "model_2_1_preprocessing_experiments.csv",
        index=False,
    )

    best_rmse = preprocessing["CV_RMSE"].min()
    tied = preprocessing[
        np.isclose(preprocessing["CV_RMSE"], best_rmse, atol=1e-10)
    ]
    selected_variant = (
        "OLS"
        if "OLS" in tied["Variant"].values
        else preprocessing.iloc[0]["Variant"]
    )

    final_model = make_pipeline(selected_variant)
    final_model.fit(
        train_data[selected_features],
        train_data["Goal_Difference"],
    )

    train_predictions = final_model.predict(train_data[selected_features])
    test_predictions = final_model.predict(test_data[selected_features])

    train_r2 = r2_score(train_data["Goal_Difference"], train_predictions)
    test_mae = mean_absolute_error(test_data["Goal_Difference"], test_predictions)
    test_mse = mean_squared_error(test_data["Goal_Difference"], test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(test_data["Goal_Difference"], test_predictions)

    response_range = (
        test_data["Goal_Difference"].max()
        - test_data["Goal_Difference"].min()
    )

    selected_cv = preprocessing[
        preprocessing["Variant"] == selected_variant
    ].iloc[0]

    baseline_scores = mean_baseline_results(
        train_data,
        test_data,
        selected_features,
    )

    baseline_comparison = pd.DataFrame(
        [
            {
                "Model": "Training-mean baseline",
                **baseline_scores,
            },
            {
                "Model": "Final linear regression",
                "CV_MAE": selected_cv["CV_MAE"],
                "CV_RMSE": selected_cv["CV_RMSE"],
                "CV_R2": selected_cv["CV_R2"],
                "Holdout_MAE": test_mae,
                "Holdout_RMSE": test_rmse,
                "Holdout_R2": test_r2,
            },
        ]
    )
    baseline_comparison.to_csv(
        RESULTS_DIR / "model_2_1_mean_baseline_vs_final.csv",
        index=False,
    )

    metrics = pd.DataFrame(
        [
            {
                "Model": "Goal Difference Model",
                "Selected_Feature_Set": selected_name,
                "Preprocessing": selected_variant,
                "Predictor_Count": len(selected_features),
                "Training_Rows": len(train_data),
                "Test_Rows": len(test_data),
                "CV_MAE": selected_cv["CV_MAE"],
                "CV_RMSE": selected_cv["CV_RMSE"],
                "CV_R2": selected_cv["CV_R2"],
                "MAE": test_mae,
                "MSE": test_mse,
                "RMSE": test_rmse,
                "NRMSE": test_rmse / response_range,
                "Test_R2": test_r2,
                "Train_R2": train_r2,
                "Train_Adjusted_R2": adjusted_r2(
                    train_r2,
                    len(train_data),
                    len(selected_features),
                ),
            }
        ]
    )

    metrics.to_csv(
        RESULTS_DIR / "model_2_1_final_metrics.csv",
        index=False,
    )

    final_dataset = model_data[
        ["Match_ID", "Goal_Difference", *selected_features]
    ].copy()

    if final_dataset.shape != (104, 10):
        raise ValueError(
            "Final Model 2.1 dataset must contain 104 rows, one ID, one response and eight predictors."
        )

    final_dataset.to_csv(
        PROCESSED_DIR / "model_2_1_final_104.csv",
        index=False,
    )

    predictions = test_data[
        ["Match_ID", "Date", "Round", "Team_A", "Team_B", "Goal_Difference"]
    ].copy()
    predictions["Predicted_Goal_Difference"] = test_predictions
    predictions["Residual"] = (
        predictions["Goal_Difference"]
        - predictions["Predicted_Goal_Difference"]
    )
    predictions.to_csv(
        RESULTS_DIR / "model_2_1_final_predictions.csv",
        index=False,
    )

    save_linearity_plots(train_data, selected_features)
    save_coefficients_and_equation(
        final_model,
        selected_features,
        selected_variant,
    )

    plt.figure(figsize=(7, 5))
    plt.scatter(test_data["Goal_Difference"], test_predictions)
    lower = min(test_data["Goal_Difference"].min(), test_predictions.min())
    upper = max(test_data["Goal_Difference"].max(), test_predictions.max())
    plt.plot([lower, upper], [lower, upper])
    plt.xlabel("Actual goal difference")
    plt.ylabel("Predicted goal difference")
    plt.title("Model 2.1: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "model_2_1_actual_vs_predicted.png",
        dpi=180,
    )
    plt.close()

    x = np.arange(len(baseline_comparison))
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(
        x - width / 2,
        baseline_comparison["CV_RMSE"],
        width,
        label="CV RMSE",
    )
    plt.bar(
        x + width / 2,
        baseline_comparison["Holdout_RMSE"],
        width,
        label="Holdout RMSE",
    )
    plt.xticks(x, baseline_comparison["Model"])
    plt.ylabel("RMSE (goal difference)")
    plt.title("Model 2.1: Mean Baseline vs Final Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "model_2_1_mean_baseline_comparison.png",
        dpi=180,
    )
    plt.close()

    residuals = test_data["Goal_Difference"].to_numpy() - test_predictions
    plt.figure(figsize=(7, 5))
    plt.scatter(test_predictions, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted goal difference")
    plt.ylabel("Residual")
    plt.title("Model 2.1: Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "model_2_1_residual_plot.png",
        dpi=180,
    )
    plt.close()

    print("\nModel 2.1 complete")
    print("Selected feature set:", selected_name)
    print("Preprocessing:", selected_variant)
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
