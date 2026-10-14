from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler


PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


CANDIDATE_FEATURES = [
    "FIFA_Points",
    "Opp_FIFA_Points",
    "FIFA_Points_Diff",
    "Prior_GF_Avg",
    "Prior_GA_Avg",
    "Prior_GD_Avg",
    "Prior_Win_Rate",
    "Prior_Draw_Rate",
    "Prior_Points_Per_Game",
    "Prior_CleanSheet_Rate",
    "Prior_FailedScore_Rate",
    "Opp_Prior_GF_Avg",
    "Opp_Prior_GA_Avg",
    "Opp_Prior_GD_Avg",
    "Opp_Prior_Win_Rate",
    "Opp_Prior_Draw_Rate",
    "Opp_Prior_Points_Per_Game",
    "Opp_Prior_CleanSheet_Rate",
    "Opp_Prior_FailedScore_Rate",
    "Rest_Days_Diff",
]


FEATURE_SETS = {
    "Initial model": [
        "FIFA_Points",
        "Opp_FIFA_Points",
        "Prior_GF_Avg",
        "Prior_GA_Avg",
        "Opp_Prior_GF_Avg",
        "Opp_Prior_GA_Avg",
        "Prior_Win_Rate",
        "Rest_Days_Diff",
    ],
    "Attack versus defence": [
        "FIFA_Points_Diff",
        "Prior_GF_Avg",
        "Opp_Prior_GA_Avg",
        "Prior_GD_Avg",
        "Opp_Prior_GD_Avg",
        "Opp_Prior_CleanSheet_Rate",
        "Prior_FailedScore_Rate",
        "Rest_Days_Diff",
    ],
    "Form and rates": [
        "FIFA_Points_Diff",
        "Prior_GF_Avg",
        "Opp_Prior_GA_Avg",
        "Opp_Prior_Win_Rate",
        "Prior_CleanSheet_Rate",
        "Opp_Prior_CleanSheet_Rate",
        "Prior_FailedScore_Rate",
        "Rest_Days_Diff",
    ],
    "Points and scoring": [
        "FIFA_Points_Diff",
        "Prior_GF_Avg",
        "Opp_Prior_GA_Avg",
        "Prior_Points_Per_Game",
        "Opp_Prior_Points_Per_Game",
        "Prior_FailedScore_Rate",
        "Opp_Prior_CleanSheet_Rate",
        "Rest_Days_Diff",
    ],
    "Relative form": [
        "FIFA_Points_Diff",
        "Prior_GF_Avg",
        "Opp_Prior_GA_Avg",
        "Prior_Win_Rate",
        "Opp_Prior_Win_Rate",
        "Prior_CleanSheet_Rate",
        "Opp_Prior_FailedScore_Rate",
        "Rest_Days_Diff",
    ],
    "Defensive pressure": [
        "FIFA_Points_Diff",
        "Prior_GF_Avg",
        "Opp_Prior_GA_Avg",
        "Prior_Draw_Rate",
        "Prior_FailedScore_Rate",
        "Opp_Prior_CleanSheet_Rate",
        "Opp_Prior_FailedScore_Rate",
        "Rest_Days_Diff",
    ],
}


def build_model_data():
    data = pd.read_csv(
        PROCESSED_DIR / "prematch_team_features_208.csv"
    )
    data["Date"] = pd.to_datetime(data["Date"])

    if len(data) != 208:
        raise ValueError("Model 2.2 must contain exactly 208 rows.")

    if data["Match_ID"].nunique() != 104:
        raise ValueError("Model 2.2 must contain 104 matches.")

    if not data.groupby("Match_ID").size().eq(2).all():
        raise ValueError("Each match must have exactly two team observations.")

    if data[CANDIDATE_FEATURES].isna().any().any():
        raise ValueError("Model 2.2 candidate predictors contain missing values.")

    return data


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
                "Correlation_With_Goals_Scored": values.corr(
                    train_data["Goals_Scored"]
                ),
            }
        )

    summary = pd.DataFrame(rows)
    summary["Absolute_Correlation"] = summary[
        "Correlation_With_Goals_Scored"
    ].abs()
    summary = summary.sort_values("Absolute_Correlation", ascending=False)
    summary.to_csv(
        RESULTS_DIR / "model_2_2_eda_summary.csv",
        index=False,
    )

    correlation = train_data[CANDIDATE_FEATURES].corr()
    correlation.to_csv(
        RESULTS_DIR / "model_2_2_predictor_correlations.csv"
    )

    plt.figure(figsize=(7, 5))
    plt.hist(train_data["Goals_Scored"], bins=8)
    plt.xlabel("Goals scored")
    plt.ylabel("Number of training team-match observations")
    plt.title("Model 2.2 Training Goals-Scored Distribution")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "model_2_2_target_histogram.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(17, 13))
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
    plt.title("Model 2.2 Training Predictor Correlation Matrix")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "model_2_2_correlation_heatmap.png",
        dpi=180,
    )
    plt.close()


def save_linearity_plots(train_data, features):
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for axis, feature in zip(axes, features):
        axis.scatter(train_data[feature], train_data["Goals_Scored"])
        axis.set_xlabel(feature)
        axis.set_ylabel("Goals Scored")

    figure.suptitle("Model 2.2: Final Predictors vs Goals Scored (Training Data)")
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "model_2_2_linearity_scatterplots.png",
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
    cv = GroupKFold(n_splits=5)

    scores = cross_validate(
        model,
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

    return {
        "CV_MAE": (-scores["test_MAE"]).mean(),
        "CV_RMSE": np.sqrt(-scores["test_MSE"]).mean(),
        "CV_R2": scores["test_R2"].mean(),
    }


def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def mean_baseline_results(train_data, test_data, features):
    baseline = DummyRegressor(strategy="mean")
    cv = GroupKFold(n_splits=5)

    scores = cross_validate(
        baseline,
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

    baseline.fit(
        train_data[features],
        train_data["Goals_Scored"],
    )
    predictions = baseline.predict(test_data[features])
    mse = mean_squared_error(
        test_data["Goals_Scored"],
        predictions,
    )

    return {
        "CV_MAE": (-scores["test_MAE"]).mean(),
        "CV_RMSE": np.sqrt(-scores["test_MSE"]).mean(),
        "CV_R2": scores["test_R2"].mean(),
        "Holdout_MAE": mean_absolute_error(
            test_data["Goals_Scored"],
            predictions,
        ),
        "Holdout_RMSE": np.sqrt(mse),
        "Holdout_R2": r2_score(
            test_data["Goals_Scored"],
            predictions,
        ),
    }


def evaluate_holdout(train_data, test_data, features, variant):
    model = make_pipeline(variant)
    model.fit(train_data[features], train_data["Goals_Scored"])

    train_predictions = model.predict(train_data[features])
    test_predictions = model.predict(test_data[features])

    mae = mean_absolute_error(test_data["Goals_Scored"], test_predictions)
    mse = mean_squared_error(test_data["Goals_Scored"], test_predictions)
    rmse = np.sqrt(mse)
    test_r2 = r2_score(test_data["Goals_Scored"], test_predictions)
    train_r2 = r2_score(train_data["Goals_Scored"], train_predictions)

    response_range = (
        test_data["Goals_Scored"].max()
        - test_data["Goals_Scored"].min()
    )

    return (
        model,
        test_predictions,
        {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "NRMSE": rmse / response_range,
            "Test_R2": test_r2,
            "Train_R2": train_r2,
            "Train_Adjusted_R2": adjusted_r2(
                train_r2,
                len(train_data),
                len(features),
            ),
        },
    )


def save_coefficients_and_equation(final_model, selected_features, selected_variant):
    regression = final_model.named_steps["model"]

    coefficients = pd.DataFrame(
        {
            "Feature": selected_features,
            "Coefficient": regression.coef_,
        }
    )
    coefficients.to_csv(
        RESULTS_DIR / "model_2_2_coefficients.csv",
        index=False,
    )

    equation_parts = [f"{regression.intercept_:.4f}"]
    for feature, coefficient in zip(selected_features, regression.coef_):
        sign = "+" if coefficient >= 0 else "-"
        equation_parts.append(
            f" {sign} {abs(coefficient):.4f}({feature})"
        )

    equation = "Predicted Goals Scored = " + "".join(equation_parts)
    (RESULTS_DIR / "model_2_2_regression_equation.txt").write_text(
        equation,
        encoding="utf-8",
    )

    print("\nFinal regression equation:")
    print(equation)

    if selected_variant != "OLS":
        print("Note: coefficients are based on transformed predictors.")


def main():
    data = build_model_data()

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
        RESULTS_DIR / "model_2_2_model_experiments.csv",
        index=False,
    )

    eligible = experiments[
        experiments["Eligible_After_Multicollinearity_Check"]
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
        RESULTS_DIR / "model_2_2_preprocessing_experiments.csv",
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

    final_model, test_predictions, final_scores = evaluate_holdout(
        train_data,
        test_data,
        selected_features,
        selected_variant,
    )

    _, initial_predictions, initial_scores = evaluate_holdout(
        train_data,
        test_data,
        FEATURE_SETS["Initial model"],
        "OLS",
    )

    selected_cv = experiments[
        experiments["Model"] == selected_name
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
                "Holdout_MAE": final_scores["MAE"],
                "Holdout_RMSE": final_scores["RMSE"],
                "Holdout_R2": final_scores["Test_R2"],
            },
        ]
    )
    baseline_comparison.to_csv(
        RESULTS_DIR / "model_2_2_mean_baseline_vs_final.csv",
        index=False,
    )

    initial_cv = experiments[
        experiments["Model"] == "Initial model"
    ].iloc[0]

    comparison = pd.DataFrame(
        [
            {
                "Model": "Initial model",
                "CV_MAE": initial_cv["CV_MAE"],
                "CV_RMSE": initial_cv["CV_RMSE"],
                "CV_R2": initial_cv["CV_R2"],
                "Holdout_MAE": initial_scores["MAE"],
                "Holdout_RMSE": initial_scores["RMSE"],
                "Holdout_R2": initial_scores["Test_R2"],
            },
            {
                "Model": "CV-selected model",
                "CV_MAE": selected_cv["CV_MAE"],
                "CV_RMSE": selected_cv["CV_RMSE"],
                "CV_R2": selected_cv["CV_R2"],
                "Holdout_MAE": final_scores["MAE"],
                "Holdout_RMSE": final_scores["RMSE"],
                "Holdout_R2": final_scores["Test_R2"],
            },
        ]
    )

    comparison.to_csv(
        RESULTS_DIR / "model_2_2_initial_vs_final.csv",
        index=False,
    )

    metrics = pd.DataFrame(
        [
            {
                "Model": "Goals Scored Model",
                "Selected_Feature_Set": selected_name,
                "Preprocessing": selected_variant,
                "Predictor_Count": len(selected_features),
                "Training_Rows": len(train_data),
                "Training_Matches": train_data["Match_ID"].nunique(),
                "Test_Rows": len(test_data),
                "Test_Matches": test_data["Match_ID"].nunique(),
                "CV_MAE": selected_cv["CV_MAE"],
                "CV_RMSE": selected_cv["CV_RMSE"],
                "CV_R2": selected_cv["CV_R2"],
                **final_scores,
            }
        ]
    )

    metrics.to_csv(
        RESULTS_DIR / "model_2_2_final_metrics.csv",
        index=False,
    )

    output_data = data.sort_values(
        ["Date", "Match_ID", "Team"]
    ).reset_index(drop=True)
    output_data.insert(
        0,
        "Observation_ID",
        range(1, len(output_data) + 1),
    )

    final_dataset = output_data[
        ["Observation_ID", "Goals_Scored", *selected_features]
    ].copy()

    if final_dataset.shape != (208, 10):
        raise ValueError(
            "Final Model 2.2 dataset must contain 208 rows, one ID, one response and eight predictors."
        )

    final_dataset.to_csv(
        PROCESSED_DIR / "model_2_2_final_208.csv",
        index=False,
    )

    predictions = test_data[
        ["Match_ID", "Date", "Round", "Team", "Opponent", "Goals_Scored"]
    ].copy()
    predictions["Predicted_Goals"] = test_predictions
    predictions["Residual"] = (
        predictions["Goals_Scored"] - predictions["Predicted_Goals"]
    )
    predictions.to_csv(
        RESULTS_DIR / "model_2_2_final_predictions.csv",
        index=False,
    )

    save_linearity_plots(train_data, selected_features)
    save_coefficients_and_equation(
        final_model,
        selected_features,
        selected_variant,
    )

    plt.figure(figsize=(7, 5))
    plt.scatter(test_data["Goals_Scored"], test_predictions)
    lower = min(test_data["Goals_Scored"].min(), test_predictions.min())
    upper = max(test_data["Goals_Scored"].max(), test_predictions.max())
    plt.plot([lower, upper], [lower, upper])
    plt.xlabel("Actual goals scored")
    plt.ylabel("Predicted goals scored")
    plt.title("Model 2.2: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "model_2_2_actual_vs_predicted.png",
        dpi=180,
    )
    plt.close()

    residuals = test_data["Goals_Scored"].to_numpy() - test_predictions
    plt.figure(figsize=(7, 5))
    plt.scatter(test_predictions, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted goals scored")
    plt.ylabel("Residual")
    plt.title("Model 2.2: Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "model_2_2_residual_plot.png",
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
        label="Grouped CV RMSE",
    )
    plt.bar(
        x + width / 2,
        baseline_comparison["Holdout_RMSE"],
        width,
        label="Holdout RMSE",
    )
    plt.xticks(x, baseline_comparison["Model"])
    plt.ylabel("RMSE (goals)")
    plt.title("Model 2.2: Mean Baseline vs Final Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "model_2_2_model_comparison.png",
        dpi=180,
    )
    plt.close()

    print("Model 2.2 complete")
    print("Selected feature set:", selected_name)
    print("Preprocessing:", selected_variant)
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
