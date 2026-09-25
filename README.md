# Zepto Data & AI Platform — Capstone Project

## Overview

This repository contains the complete Zepto Data & AI Platform capstone project with three modules:

| Module | Project | Main Technologies |
|---|---|---|
| Module 1 | Data Pipeline | Python, Requests, BeautifulSoup, Pandas, SQLite, SQL |
| Module 2 | Analytics & Machine Learning | Python, Pandas, NumPy, Seaborn, Scikit-learn, SMOTE, Joblib |
| Module 3 | GenAI Support Assistant | Sentence Transformers, ChromaDB, LangGraph, FastAPI, Pydantic, Docker |

Overall flow:

```text
DATA COLLECTION → DATA CLEANING → ANALYTICS → MACHINE LEARNING → RAG SUPPORT ASSISTANT → API
```

## Complete Repository Structure

```text
Capstone-Project/
│
├── README.md
│
├── data_pipeline/
│   ├── .gitkeep
│   ├── Module_1.ipynb
│   ├── README.md
│   ├── requirements.txt
│   ├── books.db
│   ├── cleaned_books.csv
│   └── raw_books.csv
│
├── analytics/
│   ├── Module_2.ipynb
│   ├── README.md
│   ├── requirements.txt
│   ├── class_imbalance_comparison.csv
│   ├── fare_regression_predictions.csv
│   ├── fare_regression_results.csv
│   ├── final_model_summary.csv
│   ├── final_titanic_pipeline.joblib
│   ├── model_comparison_baseline.csv
│   ├── random_forest_tuning_results.csv
│   └── titanic.csv
│
└── support_assistant/
    ├── chroma_db/
    ├── docs/
    │   ├── doc_01_delivery.txt
    │   ├── doc_02_returns.txt
    │   ├── doc_03_membership.txt
    │   ├── doc_04_orders.txt
    │   ├── doc_05_order_cancel.txt
    │   ├── doc_06_damaged_items.txt
    │   ├── doc_07_gifts.txt
    │   └── doc_08_customer.txt
    ├── .gitignore
    ├── Dockerfile
    ├── Module_3.ipynb
    ├── README.md
    ├── main.py
    └── requirements.txt
```

> **Important:** The final required names are `analytics/requirements.txt` and `analytics/titanic.csv`. If Windows currently displays `requirements.txt.txt` or `titanic (1).csv`, rename them before submission.

---

# Module 1 — Data Pipeline

## Objective

Scrape book data programmatically, clean it, convert GBP prices to INR, store the results in SQLite, and demonstrate SQL and Pandas analysis.

### Workflow

```text
Books to Scrape
      ↓
Requests + BeautifulSoup
      ↓
Pagination + Category Scraping
      ↓
Cleaning + Type Conversion
      ↓
GBP → INR
      ↓
SQLite
      ↓
SQL + Pandas
```

### Completed Work

- 93 books collected.
- 3 categories collected: Mystery, Historical Fiction, Romance.
- `price_gbp` is numeric.
- `rating` is an integer from 1–5.
- `in_stock` is Boolean.
- `price_inr` uses the required fixed rate:

```text
1 GBP = 105.50 INR
```

- SQLite contains `categories` and `books` tables with PK/FK relationship.
- SQL examples cover filtering, sorting, limiting, distinct values, range filtering, grouping/counting, and JOIN.
- SQL JOIN and Pandas `merge()` results are compared.

### Files

```text
data_pipeline/
├── Module_1.ipynb
├── README.md
├── requirements.txt
├── books.db
├── cleaned_books.csv
└── raw_books.csv
```

### Run

```powershell
cd data_pipeline
pip install -r requirements.txt
```

Run `Module_1.ipynb` from beginning to end.

---

# Module 2 — Analytics & Machine Learning

## Objective

Perform Titanic EDA, leakage-safe preprocessing, classification, imbalance comparison, Random Forest tuning, fare regression, and final pipeline persistence.

### Workflow

```text
Titanic Dataset
      ↓
EDA
      ↓
Missing Values + Outliers
      ↓
Bivariate + Correlation Analysis
      ↓
Stratified Train/Test Split
      ↓
Training-Only Preprocessing
      ↓
Classification
      ↓
Imbalance Comparison
      ↓
Random Forest GridSearchCV
      ↓
Fare Regression
      ↓
Final Joblib Pipeline
```

### Completed Work

- Titanic data loaded once and saved as `titanic.csv`.
- Missing-value percentages and percentage-based handling documented.
- IQR outlier analysis for `age` and `fare`.
- Fare mean, median, mode, and skewness comparison.
- Survival rates by sex, pclass, and sex+pclass.
- Required six-variable correlation analysis.
- Multivariate charts with written interpretations.
- Standardization before/after check.
- Logistic Regression, Decision Tree, and Random Forest.
- Confusion matrix, accuracy, precision, recall, F1, and ROC-AUC.
- Baseline vs balanced class-weight vs SMOTE comparison.
- SMOTE restricted to training data.
- Random Forest `GridSearchCV` and OOB score.
- Fare regression with MAE, RMSE, R², and Adjusted R².
- Heteroscedasticity conclusion.
- Complete fitted preprocessing + estimator pipeline saved and reloaded with Joblib.

### Files

```text
analytics/
├── Module_2.ipynb
├── README.md
├── requirements.txt
├── class_imbalance_comparison.csv
├── fare_regression_predictions.csv
├── fare_regression_results.csv
├── final_model_summary.csv
├── final_titanic_pipeline.joblib
├── model_comparison_baseline.csv
├── random_forest_tuning_results.csv
└── titanic.csv
```

### Run

```powershell
cd analytics
pip install -r requirements.txt
```

Run `Module_2.ipynb` from beginning to end.

---

# Module 3 — GenAI Support Assistant

## Objective

Build a retrieval-grounded Zepto Support Assistant using eight policy documents, local embeddings, ChromaDB, LangGraph, structured output, FastAPI, and Docker.

### Architecture

```text
8 Policy Documents
        ↓
Chunking
        ↓
all-MiniLM-L6-v2 Embeddings
        ↓
ChromaDB
        ↓
User Question
        ↓
classify_intent
   ┌────┴────┐
   ↓         ↓
Policy     General
   ↓         ↓
retrieve_   direct_
and_answer  answer
   ↓
Pydantic Response
   ↓
FastAPI /ask
```

### Policy Corpus

The eight documents are:

1. `doc_01_delivery.txt`
2. `doc_02_returns.txt`
3. `doc_03_membership.txt`
4. `doc_04_orders.txt`
5. `doc_05_order_cancel.txt`
6. `doc_06_damaged_items.txt`
7. `doc_07_gifts.txt`
8. `doc_08_customer.txt`

### Completed Work

- Eight policy documents embedded and queryable.
- `all-MiniLM-L6-v2` used for local embeddings.
- ChromaDB collection: `zepto_policies`.
- Real retrieval implemented.
- LangGraph nodes:
  - `classify_intent`
  - `retrieve_and_answer`
  - `direct_answer`
- Conditional routing implemented.
- Structured prompt includes role, context, task, format, length, negative constraint, and few-shot example.
- Default `MOCK_LLM=1` is deterministic and requires no paid API.
- Pydantic response contains `answer`, `sources`, and `confidence`.
- FastAPI `/ask` endpoint implemented.
- Dockerfile included.

### Run

```powershell
cd support_assistant
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

Open:

```text
http://localhost:7860/docs
```

### Docker

```powershell
docker build -t zepto-support-assistant .
docker run --name zepto-support -p 7860:7860 zepto-support-assistant
```

---

# Requirements Files

## Module 1

`data_pipeline/requirements.txt`

```text
pandas
requests
beautifulsoup4
jupyter
```

## Module 2

`analytics/requirements.txt`

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
jupyter
```

## Module 3

`support_assistant/requirements.txt`

```text
fastapi
uvicorn
pydantic
chromadb
sentence-transformers
langgraph
```

---

# Git Workflow

The repository includes the required Git workflow:

- Feature branch created.
- Feature branch received at least two commits.
- Work merged back into `main`.
- History can be inspected with:

```powershell
git log --graph --all --oneline
```

# Final Submission Checklist

- [x] One public GitHub repository.
- [x] `data_pipeline/`
- [x] `analytics/`
- [x] `support_assistant/`
- [x] Root `README.md`.
- [x] Module README files.
- [x] Requirements files for all three modules.
- [x] Module 1 database and CSV artifacts.
- [x] Module 2 Titanic CSV and ML artifacts.
- [x] Module 3 eight policy documents.
- [x] Module 3 ChromaDB, FastAPI, and Docker files.
- [x] Required Git workflow completed.

