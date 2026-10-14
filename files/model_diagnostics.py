from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def residual_summary(model_name, actual, predicted):
    residuals = actual - predicted
    q1 = residuals.quantile(0.25)
    q3 = residuals.quantile(0.75)
    iqr = q3 - q1

    outliers = (
        (residuals < q1 - 1.5 * iqr)
        | (residuals > q3 + 1.5 * iqr)
    ).sum()

    return {
        "Model": model_name,
        "Observations": len(actual),
        "Residual_Mean": residuals.mean(),
        "Residual_Median": residuals.median(),
        "Residual_SD": residuals.std(),
        "Residual_Min": residuals.min(),
        "Residual_Max": residuals.max(),
        "Residual_IQR_Outliers": int(outliers),
    }


def model_2_1_diagnostics():
    predictions = pd.read_csv(
        RESULTS_DIR / "model_2_1_final_predictions.csv"
    )

    actual = predictions["Goal_Difference"]
    predicted = predictions["Predicted_Goal_Difference"]
    residuals = predictions["Residual"]

    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=8)
    plt.xlabel("Residual")
    plt.ylabel("Number of matches")
    plt.title("Model 2.1 residual distribution")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "model_2_1_residual_distribution.png",
        dpi=180,
    )
    plt.close()

    outside_range = (
        (predicted < actual.min())
        | (predicted > actual.max())
    ).sum()

    summary = residual_summary(
        "Model 2.1",
        actual,
        predicted,
    )
    summary["Negative_Predictions"] = "N/A"
    summary["Predictions_Outside_Holdout_Range"] = int(outside_range)

    return summary


def model_2_2_diagnostics():
    predictions = pd.read_csv(
        RESULTS_DIR / "model_2_2_final_predictions.csv"
    )

    actual = predictions["Goals_Scored"]
    predicted = predictions["Predicted_Goals"]
    residuals = predictions["Residual"]

    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=8)
    plt.xlabel("Residual")
    plt.ylabel("Number of team-match observations")
    plt.title("Model 2.2 residual distribution")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "model_2_2_residual_distribution.png",
        dpi=180,
    )
    plt.close()

    outside_range = (
        (predicted < actual.min())
        | (predicted > actual.max())
    ).sum()

    summary = residual_summary(
        "Model 2.2",
        actual,
        predicted,
    )
    summary["Negative_Predictions"] = int((predicted < 0).sum())
    summary["Predictions_Outside_Holdout_Range"] = int(outside_range)

    return summary


def main():
    diagnostics = pd.DataFrame(
        [
            model_2_1_diagnostics(),
            model_2_2_diagnostics(),
        ]
    )

    diagnostics.to_csv(
        RESULTS_DIR / "final_model_diagnostics_summary.csv",
        index=False,
    )

    print("Final diagnostics complete")
    print()
    print("Residual diagnostics:")
    print(diagnostics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
