# Capstone Project - Zepto Data & AI Platform

## Module 2 - Analytics and Machine Learning

**Module:** 2  
**Weightage:** 50 Marks  
**Focus:** Exploratory Data Analysis, Classification, Class Imbalance, Hyperparameter Tuning, Regression, and Machine Learning Pipeline

---

## 1. Project Overview

Module 2 focuses on transforming data into analytical insights and Machine Learning models using the Titanic dataset.

The module covers:

- Data understanding and cleaning
- Exploratory Data Analysis (EDA)
- Machine Learning data preparation
- Classification
- Model comparison
- Class imbalance handling
- SMOTE
- Random Forest hyperparameter tuning
- Out-of-Bag (OOB) evaluation
- Fare regression
- Model evaluation
- Complete Machine Learning pipeline
- Model saving and reloading with `joblib`

### Overall Workflow

```text
Data
  |
  v
Data Understanding
  |
  v
Data Exploration
  |
  v
Data Preparation
  |
  v
Classification Models
  |
  v
Model Comparison
  |
  v
Class Imbalance Handling
  |
  v
Random Forest Tuning
  |
  v
Fare Regression
  |
  v
Final Machine Learning Pipeline
  |
  v
Save and Reload Model
```

---

## 2. Objectives

The main objectives of this module are:

1. Understand and inspect the Titanic dataset.
2. Handle missing values.
3. Perform Exploratory Data Analysis.
4. Prepare features and target variables.
5. Build classification models.
6. Evaluate and compare classification models.
7. Handle class imbalance.
8. Apply SMOTE only to training data.
9. Tune Random Forest using GridSearchCV.
10. Report the best parameters and OOB score.
11. Predict Fare using Multivariate Linear Regression.
12. Evaluate regression using MAE, RMSE, R², and Adjusted R².
13. Analyze regression residuals.
14. Build a complete preprocessing and model pipeline.
15. Save the final pipeline using `joblib`.
16. Reload the pipeline and test it using raw passenger input.

---

## 3. Dataset

The Titanic dataset is used for the Machine Learning tasks.

Important variables include:

- Passenger class
- Sex
- Age
- Number of siblings and spouses
- Number of parents and children
- Fare
- Embarkation port
- Survival status

### Classification Target

```text
survived
```

### Regression Target

```text
fare
```

---

## 4. Data Understanding

The dataset was inspected using:

- Dataset shape
- Column names
- Data types
- First few records
- Statistical summary
- Missing-value counts
- Missing-value percentages

Example:

```python
df.shape
df.columns
df.dtypes
df.head()
df.describe()
df.isnull().sum()
```

---

## 5. Missing Value Handling

Missing values were analyzed before Machine Learning.

The following approach was used:

- `deck` was removed because of its high percentage of missing values.
- Missing `age` values were replaced using the median.
- Missing `embarked` values were replaced using the mode.
- Missing `embark_town` values were replaced using the mode.

Example:

```python
df = df.drop(columns=["deck"])

df["age"] = df["age"].fillna(df["age"].median())

df["embarked"] = df["embarked"].fillna(
    df["embarked"].mode()[0]
)

df["embark_town"] = df["embark_town"].fillna(
    df["embark_town"].mode()[0]
)
```

A final missing-value check was performed after cleaning.

---

## 6. Exploratory Data Analysis

Four major visualizations were created.

### 6.1 Age Distribution

A histogram was used to understand passenger age distribution.

```python
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="age",
    bins=30,
    kde=True
)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.show()
```

**Observation:** The distribution contains more passengers in younger and middle-age ranges, with fewer passengers at older ages.

### 6.2 Fare Distribution

A box plot was used to understand Fare distribution and identify potential outliers.

```python
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="fare"
)

plt.title("Fare Distribution of Titanic Passengers")
plt.xlabel("Fare")

plt.show()
```

**Observation:** Most fares are concentrated at lower values, with several higher-fare observations appearing as potential outliers.

### 6.3 Survival Comparison by Gender

A count plot was used to compare survival outcomes across genders.

```python
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="sex",
    hue="survived"
)

plt.title("Survival Comparison by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.show()
```

**Observation:** The chart shows different survival distributions across the gender categories.

This describes an association in the dataset and does not establish causation.

### 6.4 Correlation Heatmap

A correlation heatmap was created for numerical variables.

```python
plt.figure(figsize=(10, 7))

numeric_df = df.select_dtypes(include="number")

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Titanic Numerical Variables")

plt.show()
```

**Observation:** The heatmap helps identify linear relationships between numerical variables.

Correlation represents association and should not be interpreted as proof of causation.

---

## 7. Machine Learning Data Preparation

### 7.1 Classification Target

```python
y = df["survived"]
```

### 7.2 Classification Features

The following columns were removed from the initial feature set:

```text
survived
alive
class
embark_town
who
adult_male
```

The resulting features include:

```text
pclass
sex
age
sibsp
parch
fare
embarked
alone
```

`alive` was removed to avoid direct target leakage because it represents the survival outcome in another form.

---

## 8. Train and Test Split

The data was divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The split contains:

```text
80% - Training data
20% - Testing data
```

Stratification was used to preserve the target-class distribution.

---

## 9. Data Preprocessing

### Numerical Features

```text
pclass
age
sibsp
parch
fare
```

Numerical features were scaled using `StandardScaler`.

### Categorical Features

```text
sex
embarked
alone
```

Categorical features were encoded using `OneHotEncoder`.

A `ColumnTransformer` was used to combine both preprocessing operations.

The preprocessing was included inside the Machine Learning pipelines to reduce the risk of data leakage.

---

## 10. Classification Models

Three baseline classification models were trained.

### Logistic Regression

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

### Decision Tree

```python
DecisionTreeClassifier(
    random_state=42
)
```

### Random Forest

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

---

## 11. Classification Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC

Confusion matrices and ROC curves were also created.

Baseline results were saved as:

```text
model_comparison_baseline.csv
```

---

## 12. Class Imbalance Handling

The target-class distribution was checked before applying imbalance-handling techniques.

Three Random Forest approaches were compared.

### Baseline Random Forest

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

### Balanced Random Forest

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)
```

### SMOTE Random Forest

SMOTE was used to generate synthetic minority-class samples.

```python
SMOTE(random_state=42)
```

SMOTE was applied only to the training data. The test data remained untouched.

The approaches were compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC

Results were saved as:

```text
class_imbalance_comparison.csv
```

---

## 13. Random Forest Hyperparameter Tuning

GridSearchCV was used to tune the Random Forest model.

The following parameters were tuned:

```text
n_estimators
max_depth
max_features
```

Parameter grid:

```python
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"]
}
```

Five-fold cross-validation was used with F1 Score as the scoring metric.

The best parameters and cross-validation score were recorded.

---

## 14. Random Forest OOB Evaluation

The tuned Random Forest was configured with:

```python
oob_score=True
```

The OOB score was retrieved from the best Random Forest model.

The tuning results include:

- Best `n_estimators`
- Best `max_depth`
- Best `max_features`
- Cross-validation F1
- Test Accuracy
- Test Precision
- Test Recall
- Test F1
- Test AUC
- OOB Score

Results were saved as:

```text
random_forest_tuning_results.csv
```

---

## 15. Fare Regression

A separate regression task was created to predict:

```text
fare
```

Multivariate Linear Regression was used.

Since `fare` is the target, it was removed from the regression input features.

The regression workflow was:

```text
Raw Features
     |
     v
Preprocessing
     |
     v
Linear Regression
     |
     v
Predicted Fare
```

---

## 16. Regression Evaluation

The regression model was evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute prediction error.

### Root Mean Squared Error (RMSE)

Gives greater weight to larger prediction errors.

### R²

Measures the proportion of variation explained by the regression model.

### Adjusted R²

Accounts for the number of predictors used by the model.

Results were saved as:

```text
fare_regression_results.csv
```

Predictions were saved as:

```text
fare_regression_predictions.csv
```

---

## 17. Residual Analysis

Residuals were calculated as:

```text
Residual = Actual Fare - Predicted Fare
```

A residual plot was created with:

- Predicted Fare on the x-axis
- Residuals on the y-axis
- A horizontal reference line at zero

The residual plot was used to inspect the pattern of prediction errors.

---

## 18. Complete Machine Learning Pipeline

The final classification pipeline combines preprocessing and the final Random Forest model.

```text
Raw Input
    |
    v
Preprocessing
    |
    +-- Numerical Scaling
    |
    +-- Categorical Encoding
    |
    v
Final Random Forest Model
    |
    v
Prediction
```

The complete preprocessing and model pipeline was saved together.

This allows new raw data to be passed directly to the saved pipeline without manually performing scaling and encoding again.

---

## 19. Model Saving

The final pipeline was saved using `joblib`.

```python
import joblib

joblib.dump(
    final_pipeline,
    "final_titanic_pipeline.joblib"
)
```

Saved model:

```text
final_titanic_pipeline.joblib
```

The saved file contains:

```text
Preprocessing + Model
```

---

## 20. Model Reloading

The saved pipeline can be loaded again:

```python
loaded_pipeline = joblib.load(
    "final_titanic_pipeline.joblib"
)
```

The reloaded pipeline can then be used for prediction.

---

## 21. Testing with Raw Input

Example raw passenger input:

```python
sample_passenger = pd.DataFrame([{
    "pclass": 3,
    "sex": "male",
    "age": 30,
    "sibsp": 0,
    "parch": 0,
    "fare": 20.0,
    "embarked": "S",
    "alone": True
}])
```

Prediction:

```python
prediction = loaded_pipeline.predict(
    sample_passenger
)
```

Prediction interpretation:

```text
0 - Did not survive
1 - Survived
```

Prediction probabilities can also be obtained using:

```python
loaded_pipeline.predict_proba(
    sample_passenger
)
```

---

## 22. Project Structure

Suggested Module 2 structure:

```text
Module-2/
|
+-- Module_2_Analytics_ML.ipynb
+-- README.md
|
+-- model_comparison_baseline.csv
+-- class_imbalance_comparison.csv
+-- random_forest_tuning_results.csv
+-- fare_regression_results.csv
+-- fare_regression_predictions.csv
+-- final_model_summary.csv
|
+-- final_titanic_pipeline.joblib
```

---

## 23. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Joblib
- Google Colab / Jupyter Notebook
- GitHub

---

## 24. Key Machine Learning Concepts Demonstrated

- Data cleaning
- Missing-value handling
- Exploratory Data Analysis
- Feature preparation
- Data leakage prevention
- Train/test splitting
- Feature scaling
- One-hot encoding
- Machine Learning pipelines
- Logistic Regression
- Decision Tree
- Random Forest
- Classification metrics
- Confusion Matrix
- ROC Curve
- AUC
- Class imbalance
- `class_weight="balanced"`
- SMOTE
- Cross-validation
- GridSearchCV
- Hyperparameter tuning
- OOB evaluation
- Linear Regression
- MAE
- RMSE
- R²
- Adjusted R²
- Residual analysis
- Model serialization
- Model reloading
- Prediction using raw input

---

## 25. Generated Output Files

| File | Description |
|---|---|
| `model_comparison_baseline.csv` | Comparison of baseline classification models |
| `class_imbalance_comparison.csv` | Comparison of baseline, balanced, and SMOTE Random Forest |
| `random_forest_tuning_results.csv` | GridSearchCV and OOB results |
| `fare_regression_results.csv` | Fare regression evaluation metrics |
| `fare_regression_predictions.csv` | Actual vs predicted Fare values |
| `final_model_summary.csv` | Final classification model summary |
| `final_titanic_pipeline.joblib` | Saved preprocessing and final model pipeline |

---

## 26. Module 2 Workflow

```text
Data Understanding
       |
       v
Missing Value Handling
       |
       v
Exploratory Data Analysis
       |
       v
Feature Preparation
       |
       v
Train/Test Split
       |
       v
Preprocessing
       |
       +-----------------------------+
       |             |               |
       v             v               v
Logistic        Decision Tree    Random Forest
Regression
       |             |               |
       +-------------+---------------+
                     |
                     v
              Model Comparison
                     |
                     v
             Class Imbalance
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Baseline   Balanced    SMOTE
          |          |          |
          +----------+----------+
                     |
                     v
               GridSearchCV
                     |
                     v
            Tuned Random Forest
                     |
                     v
               Fare Regression
                     |
                     v
            Final ML Pipeline
                     |
                     v
                Save Model
                     |
                     v
               Reload Model
                     |
                     v
                Raw Input
                     |
                     v
                Prediction
```

---

## 27. Complete Capstone Structure

This module is part of the complete Capstone Project.

```text
CAPSTONE PROJECT
|
+-- MODULE 1
|   +-- Data Pipeline
|
+-- MODULE 2
|   +-- Analytics and Machine Learning
|
+-- MODULE 3
    +-- GenAI Support Assistant
```

Overall project flow:

```text
Get Data
   |
   v
Understand Data
   |
   v
Build Intelligence
   |
   v
Serve It
```

## 29. Project Status

**Module 2 - Analytics and Machine Learning**

Status: **Completed**

