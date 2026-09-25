# Capstone Project — Zepto Data & AI Platform

## Certificate Program in Artificial Intelligence and Machine Learning

This repository contains my complete Capstone Project with three connected modules.

**GET DATA → UNDERSTAND DATA → BUILD INTELLIGENCE → SERVE IT**

The three modules are part of one connected engineering story: data collection, analytics and machine learning, followed by a retrieval-grounded GenAI application.

## Project Overview

| Module | Project | Marks | Main Technologies |
|---|---|---:|---|
| Module 1 | Data Pipeline | 25 | Python, Requests, BeautifulSoup, SQLite, SQL, Pandas |
| Module 2 | Analytics + Machine Learning | 50 | Python, Pandas, Seaborn, Scikit-learn, Joblib |
| Module 3 | GenAI Support Assistant | 25 | Sentence Transformers, ChromaDB, LangGraph, FastAPI, Pydantic, Docker |

**Total: 100 Marks**

---

# Module 1 — Data Pipeline

## Objective

Collect catalog-style book data from a public website, clean it, store it in SQLite, and perform SQL/Pandas analysis.

### Data Source

`books.toscrape.com`

### Pipeline

**Website → Scrape → Clean → Convert → Store → Query**

### Data Collected

- Title
- Price
- Star Rating
- Availability
- Category

### Implementation

- Used `requests` and `BeautifulSoup`.
- Collected **93 books** across **3 categories**.
- Cleaned and converted scraped fields.
- Created GBP and INR price columns.
- Created SQLite `categories` and `books` tables with PK/FK relationships.
- Executed SQL queries including filtering, sorting, limiting, distinct/category operations, and JOIN.
- Used `pandas.read_sql()`.
- Reproduced the JOIN result using `pandas.merge()`.

### Files

```text
Module-1-Data-Pipeline/
├── Module_1.ipynb
├── README.md
├── books.db
└── cleaned_books.csv
```

---

# Module 2 — Analytics + Machine Learning

## Objective

Perform EDA, preprocess the Titanic dataset, train and compare classification models, evaluate class imbalance, tune Random Forest, build a regression model, and save the final preprocessing + model pipeline.

### Dataset

**Titanic dataset**

### Workflow

**Titanic Dataset → Understand → Clean → Analyze → Model → Evaluate → Improve → Save**

### EDA

- Dataset shape and structure
- Data types
- Missing values and percentages
- Data cleaning and preprocessing
- Distribution analysis
- Survival analysis
- Correlation analysis
- Meaningful visualizations

### Classification

Target:

`survived`

Models:

- Logistic Regression
- Decision Tree
- Random Forest

Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Class Imbalance

Compared:

- Baseline Random Forest
- Class-weight balancing
- SMOTE

### Hyperparameter Tuning

Used `GridSearchCV` for Random Forest.

### Regression

Predicted:

`Fare`

Metrics:

- MAE
- RMSE
- R²
- Adjusted R²

### Final Model

The complete preprocessing + model pipeline is saved as:

`final_titanic_pipeline.joblib`

The saved pipeline was reloaded and tested with raw passenger input.

### Files

```text
Module-2-Analytics-and-Machine-Learning/
├── Module_2.ipynb
├── README.md
├── class_imbalance_comparison.csv
├── fare_regression_predictions.csv
├── fare_regression_results.csv
├── final_model_summary.csv
├── final_titanic_pipeline.joblib
├── model_comparison_baseline.csv
└── random_forest_tuning_results.csv
```

---

# Module 3 — GenAI Support Assistant

## Objective

Build a GenAI support assistant that answers Zepto policy questions using document retrieval, embeddings, ChromaDB, LangGraph, structured output, FastAPI, and Docker.

The graded version works in deterministic **fully offline mock mode**, so no external LLM API is required.

### Architecture

**Policy Documents → Chunking → Embeddings → ChromaDB → Retrieval → LangGraph → Answer → FastAPI**

### Policy Documents

Eight documents cover:

1. Delivery
2. Returns & Refunds
3. Membership
4. Order Tracking
5. Cancellation
6. Damaged / Missing Items
7. Gift Cards
8. Support Hours

### Embeddings

- Library: `sentence-transformers`
- Model: `all-MiniLM-L6-v2`
- Vector store: ChromaDB
- Collection: `zepto_policies`

### Retrieval

For policy queries:

1. Embed the query.
2. Search ChromaDB.
3. Retrieve the top 3 relevant documents.
4. Generate a response using the retrieved context.

### LangGraph

```text
classify_intent
       │
       ├── Policy Query ──→ retrieve_and_answer
       │
       └── General Query ─→ direct_answer
```

### Structured Output

```json
{
  "answer": "string",
  "sources": ["string"],
  "confidence": 1.0
}
```

### Mock Mode

Default:

```text
MOCK_LLM=1
```

The required graded path works without an API key or external LLM service.

### FastAPI

Endpoint:

```text
POST /ask
```

Example request:

```json
{
  "query": "How much does delivery cost for an order below INR 149?"
}
```

### Docker

The application includes a Dockerfile and runs locally with Uvicorn.

```bash
docker build -t zepto-support-assistant .
docker run --name zepto-support -p 7860:7860 zepto-support-assistant
```

Swagger:

```text
http://localhost:7860/docs
```

### Files

```text
Module-3-GenAI-Support-Assistant/
├── chroma_db/
│   ├── chroma.sqlite3
│   └── ChromaDB index files
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

---

# End-to-End Capstone Architecture

```text
                    CAPSTONE PROJECT
                           │
                           ▼
                  ┌─────────────────┐
                  │    MODULE 1     │
                  │   Data Pipeline │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │    MODULE 2     │
                  │ Analytics + ML  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │    MODULE 3     │
                  │ GenAI Assistant │
                  └─────────────────┘
```

**DATA → ANALYSIS → ML → GENAI → API**

---

# Repository Structure

```text
Capstone-Project/
│
├── README.md
│
├── Module-1-Data-Pipeline/
│
├── Module-2-Analytics-and-Machine-Learning/
│
└── Module-3-GenAI-Support-Assistant/
```

Each module contains its implementation, outputs, and supporting files.

---

# Testing and Completion

## Module 1

- 93 books collected
- 3 categories represented
- Data cleaned and converted
- SQLite database created
- Related tables created
- SQL queries executed
- `pandas.read_sql()` used
- `pandas.merge()` used for JOIN equivalence

## Module 2

- EDA completed
- Missing values handled
- Classification models trained
- Model comparison completed
- Class imbalance evaluated
- Random Forest tuning completed
- Regression completed
- Final pipeline saved with `joblib`
- Saved pipeline reloaded and tested

## Module 3

- 8 policy documents embedded
- ChromaDB populated
- Top-3 retrieval tested
- LangGraph routing tested
- Mock mode tested
- Pydantic validation tested
- Retry logic tested
- FastAPI `/ask` tested
- Docker image built successfully
- Docker container run successfully
- Final Colab audit: **9/9 checks passed**

---

# Technologies Used

### Programming
Python, SQL

### Data Engineering
Requests, BeautifulSoup, SQLite, Pandas

### Machine Learning
NumPy, Pandas, Scikit-learn, Seaborn, Joblib, SMOTE

### Generative AI
Sentence Transformers, `all-MiniLM-L6-v2`, ChromaDB, LangGraph, Pydantic, FastAPI

### Deployment
Docker, Uvicorn

### Development
Google Colab, GitHub


## Git Workflow

This project was developed using a feature-branch workflow.

- `main` contains the final submission.
- `feature/final-submission` was used for final project updates.
- The feature branch contains multiple commits and is merged back into `main`.
- The complete Git history can be inspected using:

```bash
git log --graph --oneline --all --decorate

---

# Final Outcome

This capstone demonstrates an end-to-end AI/ML engineering workflow:

**Collect data → Clean and store data → Analyze data → Build ML models → Evaluate and save a pipeline → Build a retrieval-grounded GenAI assistant → Serve it through an API → Containerize with Docker.**

