# Module 2 — Analytics + Machine Learning

## Objective

Analyze the Titanic dataset, build leakage-safe machine-learning pipelines, compare classification approaches, tune Random Forest, perform fare regression, and save the complete final pipeline.

## Folder

```text
analytics/
```

## Dataset

The Titanic dataset is loaded once and immediately saved as:

```text
titanic.csv
```

The CSV provides the required offline copy and can be loaded with:

```python
pd.read_csv("titanic.csv")
```

## Workflow

```text
Titanic Dataset
      ↓
Profiling
      ↓
Missing-Value Analysis
      ↓
EDA
      ↓
Train/Test Split
      ↓
Training-Only Preprocessing
      ↓
Classification
      ↓
Class Imbalance
      ↓
Random Forest Tuning
      ↓
Fare Regression
      ↓
Final Pipeline
```

## Missing Values

Missing-value percentages are calculated for affected columns.

A percentage-based rule is used to decide whether a column should be dropped or imputed.

For model inputs, imputation is performed inside the preprocessing pipeline so that learned statistics are fitted only on the training data.

- Numeric variables: median imputation.
- Categorical variables: most-frequent imputation.
- `deck`, which has extensive missingness, is removed from the modeling dataset.

This prevents test-set information from leaking into training.

## EDA

The notebook includes:

- Dataset shape and structure.
- Data types and descriptive statistics.
- Missing-value percentages.
- IQR outlier counts for `age` and `fare`.
- Fare mean, median, mode, and skewness comparison.
- Numeric survival rates by:
  - sex
  - passenger class
  - sex + passenger class
- Required correlation matrix using:
  - `survived`
  - `pclass`
  - `age`
  - `sibsp`
  - `parch`
  - `fare`
- Interpretation of the two strongest absolute off-diagonal correlations.
- Four or more multivariate charts with written interpretations.
- Before/after standardization check for `age` and `fare`.

`adult_male` and `alone` are excluded from the required correlation matrix.

## Classification

Target:

```text
survived
```

Models:

- Logistic Regression
- Decision Tree
- Random Forest

Evaluation includes:

- Confusion matrix
- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

The Decision Tree is visualized with labeled feature and class names.

## Class Imbalance

Three Random Forest approaches are compared:

1. Baseline.
2. `class_weight="balanced"`.
3. SMOTE.

SMOTE is applied only within the training workflow and never to the test set.

## Hyperparameter Tuning

Random Forest is tuned with `GridSearchCV`.

The notebook reports:

- Best parameters.
- Cross-validation score.
- OOB score from a Random Forest configured with `oob_score=True`.

## Regression

The regression task predicts:

```text
fare
```

Reported metrics:

- MAE
- RMSE
- R²
- Adjusted R²

Residual analysis is used to assess heteroscedasticity, followed by an explicit written conclusion.

## Saved Model

The complete fitted pipeline containing preprocessing and the final estimator is saved as:

```text
final_titanic_pipeline.joblib
```

The notebook reloads the saved pipeline and demonstrates prediction using raw passenger data.

## How to Run

Open:

```text
Module_2.ipynb
```

Run all cells from beginning to end.

The notebook generates/uses the offline `titanic.csv` and creates the model-result artifacts.

## Expected Files

```text
analytics/
├── Module_2.ipynb
├── README.md
├── titanic.csv
├── final_titanic_pipeline.joblib
├── class_imbalance_comparison.csv
├── fare_regression_predictions.csv
├── fare_regression_results.csv
├── final_model_summary.csv
├── model_comparison_baseline.csv
└── random_forest_tuning_results.csv
```

## Design Decisions

1. The raw Titanic dataset is loaded once.
2. `titanic.csv` provides an offline fallback.
3. The classifier split is stratified.
4. Preprocessing is fitted only on training data.
5. Imputation and scaling are contained in pipelines.
6. SMOTE is restricted to the training workflow.
7. The final saved artifact contains preprocessing and the estimator together.
