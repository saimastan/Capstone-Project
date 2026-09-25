# Zepto Data & AI Platform — Capstone Project

## Overview

This repository contains the complete Zepto Data & AI Platform capstone project with three modules:

| Module | Project | Main Technologies |
|---|---|---|
| Module 1 | Data Pipeline | Python, Requests, BeautifulSoup, Pandas, SQLite, SQL |
| Module 2 | Analytics + Machine Learning | Python, Pandas, Seaborn, Scikit-learn, SMOTE, Joblib |
| Module 3 | GenAI Support Assistant | Sentence Transformers, ChromaDB, LangGraph, FastAPI, Pydantic, Docker |

The project follows the workflow:

**DATA COLLECTION → DATA ANALYSIS → MACHINE LEARNING → RETRIEVAL-AUGMENTED SUPPORT ASSISTANT → API**

## Repository Structure

```text
Capstone-Project/
├── README.md
├── data_pipeline/
│   ├── Module_1.ipynb
│   ├── README.md
│   ├── books.db
│   └── cleaned_books.csv
├── analytics/
│   ├── Module_2.ipynb
│   ├── README.md
│   ├── titanic.csv
│   └── model artifacts
└── support_assistant/
    ├── Module_3.ipynb
    ├── README.md
    ├── docs/
    ├── main.py
    ├── Dockerfile
    └── requirements.txt
```

## Dependency Approach

Each module uses its own dependency file where practical. Module 3 includes a `requirements.txt`. Module 1 and Module 2 notebooks install/import the libraries needed for their workflows.

## Module 1 — Data Pipeline

### Objective

Scrape book data, clean and convert the fields, store the data in a normalized SQLite database, and query it using SQL and Pandas.

### Workflow

```text
Books to Scrape
      ↓
Requests + BeautifulSoup
      ↓
Cleaning and Type Conversion
      ↓
GBP → INR Conversion
      ↓
SQLite Database
      ↓
SQL + Pandas Analysis
```

### Completed Work

- 93 books collected.
- 3 categories represented.
- `price_gbp` converted to float.
- `rating` converted to integer values from 1–5.
- `in_stock` represented as boolean.
- `price_inr` calculated using the required fixed rate:
  **1 GBP = 105.50 INR**.
- SQLite database created with related `categories` and `books` tables.
- Required SQL operations demonstrated, including filtering, sorting, limiting, distinct values, range filtering, grouping/counting, and JOIN.
- SQL JOIN and Pandas `merge()` results were compared.

### Run

Open:

```text
data_pipeline/Module_1.ipynb
```

Run the notebook from beginning to end.

See `data_pipeline/README.md` for module-specific details.

## Module 2 — Analytics + Machine Learning

### Objective

Perform exploratory data analysis on the Titanic dataset, build leakage-safe preprocessing pipelines, train classification models, evaluate class imbalance, tune Random Forest, perform fare regression, and save the final fitted pipeline.

### Workflow

```text
Titanic Dataset
      ↓
Profiling + EDA
      ↓
Missing-Value Analysis
      ↓
Train/Test Split
      ↓
Leakage-Safe Preprocessing
      ↓
Classification
      ↓
Imbalance Comparison
      ↓
Random Forest Tuning
      ↓
Fare Regression
      ↓
Final Pipeline
```

### Completed Work

- Titanic dataset loaded once and saved as `titanic.csv` for offline use.
- Missing-value percentages reported.
- Percentage-based missing-value handling rule documented.
- Imputation performed inside training pipelines.
- IQR outlier analysis performed for `age` and `fare`.
- Fare mean, median, mode, and skewness compared.
- Survival rates calculated by sex, passenger class, and sex + passenger class.
- Required six-variable correlation analysis performed.
- Multivariate visualizations with written interpretations included.
- Standardization before/after check included.
- Logistic Regression, Decision Tree, and Random Forest trained.
- Accuracy, precision, recall, F1, confusion matrix, and ROC-AUC evaluated.
- Baseline, balanced class-weight, and SMOTE Random Forest approaches compared.
- SMOTE applied only within the training workflow.
- Random Forest tuned using `GridSearchCV` and evaluated with OOB score.
- Fare regression evaluated using MAE, RMSE, R², and Adjusted R².
- Heteroscedasticity analysis and conclusion included.
- Final fitted preprocessing + estimator pipeline saved with Joblib and reloaded for raw-data prediction.

### Run

Open:

```text
analytics/Module_2.ipynb
```

Run the notebook from beginning to end.

See `analytics/README.md` for module-specific details and generated artifacts.

## Module 3 — GenAI Support Assistant

### Objective

Build a retrieval-grounded Zepto support assistant using a fixed policy corpus, local embeddings, ChromaDB retrieval, LangGraph routing, structured output, FastAPI, and Docker.

### Workflow

```text
8 Policy Documents
       ↓
Chunking
       ↓
all-MiniLM-L6-v2 Embeddings
       ↓
ChromaDB
       ↓
Intent Classification
       ↓
┌───────────────────────┐
│                       │
Policy Question    General Question
│                       │
↓                       ↓
Retrieval          Direct Answer
│
↓
Mock/LLM Generation
│
↓
Pydantic Validation
│
↓
FastAPI /ask
```

### Completed Work

- Eight policy documents included.
- Local `all-MiniLM-L6-v2` embeddings generated.
- ChromaDB collection `zepto_policies` created.
- Real retrieval implemented.
- LangGraph uses `classify_intent`, `retrieve_and_answer`, and `direct_answer`.
- Conditional routing implemented.
- Structured prompt contains role, context, task, format, length, negative constraint, and few-shot example.
- Default `MOCK_LLM=1` path is deterministic and does not require an external LLM API.
- Pydantic structured response contains `answer`, `sources`, and `confidence`.
- Retry validation is included for the optional LLM path.
- FastAPI `/ask` endpoint implemented.
- Dockerfile and local Docker instructions included.

### Run Locally

From `support_assistant/`:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

Open:

```text
http://localhost:7860/docs
```

The graded baseline uses:

```text
MOCK_LLM=1
```

### Docker

```bash
docker build -t zepto-support-assistant .
docker run --name zepto-support -p 7860:7860 zepto-support-assistant
```

Then open:

```text
http://localhost:7860/docs
```

See `support_assistant/README.md` for the complete module documentation.

## Git Workflow

The repository includes the required Git workflow:

- A feature branch was created.
- The feature branch received at least two commits.
- The work was merged back into `main`.
- The history can be inspected with:

```bash
git log --graph --all --oneline
```

## Final Submission Notes

- Submit exactly one public GitHub repository link.
- Keep the three required root folders exactly:
  `data_pipeline`, `analytics`, and `support_assistant`.
- Keep the root `README.md`.
- Do not rely on screenshots or presentations for required written interpretations.
- Generated chart images may be supporting artifacts, but written interpretations remain in Markdown/notebook cells.
