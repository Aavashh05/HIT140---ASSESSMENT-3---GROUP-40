# HIT140 Foundations of Data Science – Assessment 3

## Objective 2: FIFA World Cup 2026 Regression Analysis

### Group 40
- Aavash Khatiwada – S405266
- Dipak Karki – S407785
- Kamala Thapa – S407760
- Usha Khadka – S408025

## Project Overview

This project completes Objective 2 of HIT140 Assessment 3 using FIFA World Cup 2026 data.

Two multiple linear regression models were developed:

- **Regression 2.1:** Predicts goal difference using exactly 104 match observations.
- **Regression 2.2:** Predicts goals scored by a team using exactly 208 team-match observations.

Each final model contains exactly eight pre-match explanatory variables.

Historical form was created using the four most recent matches available before each target match. The target match itself was not used to create its predictors.

## Data Sources

The analysis uses:

- FIFA World Cup 2026 fixtures and results
- FBRef team schedule data
- FBRef shooting data
- FIFA rankings dated 11 June 2026

Raw datasets are stored in:

`data/raw/`

Processed datasets are stored in:

`data/processed/`

## Final Models

### Regression 2.1 – Goal Difference

Response:

`Goal_Difference = Team A Goals - Team B Goals`

Final predictors:

1. FIFA_Points_Diff
2. Prior_GF_Avg_Diff
3. Prior_GA_Avg_Diff
4. Prior_Draw_Rate_Diff
5. Prior_CleanSheet_Rate_Diff
6. Prior_FailedScore_Rate_Diff
7. Prior_Opp_FIFA_Points_Avg_Diff
8. Prior_Strength_Adjusted_GD_Avg_Diff

Final model: **Yeo-Johnson Multiple Linear Regression**

Holdout results:

- MAE = 1.3280
- RMSE = 1.7385
- R² = 0.4520

### Regression 2.2 – Goals Scored

The 104 matches were converted into 208 team-match observations.

Final predictors:

1. FIFA_Points_Diff
2. Prior_GF_Avg
3. Opp_Prior_GA_Avg
4. Prior_Draw_Rate
5. Prior_FailedScore_Rate
6. Opp_Prior_CleanSheet_Rate
7. Opp_Prior_FailedScore_Rate
8. Rest_Days_Diff

Final model: **Multiple Linear Regression (OLS)**

Holdout results:

- MAE = 0.9117
- RMSE = 1.2250
- R² = 0.3006

## Python Files

The main Python scripts are stored in the `files/` folder:

- `data_preparation.py` – prepares and combines the datasets
- `goal_difference_model.py` – performs Regression 2.1
- `goals_scored_model.py` – performs Regression 2.2
- `model_diagnostics.py` – performs residual and prediction diagnostics
- `algorithm_comparison.py` – compares Linear Regression with SVR, Random Forest and Gradient Boosting

## How to Run

Run the scripts from the project root in this order:

```bash
python files/data_preparation.py
python files/goal_difference_model.py
python files/goals_scored_model.py
python files/model_diagnostics.py
python files/algorithm_comparison.py
