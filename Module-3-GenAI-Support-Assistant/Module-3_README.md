# Module 3 — GenAI Support Assistant

## Zepto Data & AI Platform Capstone Project

### Project Overview

This module implements a GenAI-powered support assistant for Zepto policy questions.

The assistant uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from a fixed set of Zepto policy documents and return a grounded response.

The graded baseline is deterministic and runs in offline mock mode using:

```text
MOCK_LLM=1
```

No paid LLM API, API key, or external LLM service is required for the graded baseline.

---

## 1. Business Problem

Customers may ask questions about Zepto policies such as:

- Delivery
- Returns & Refunds
- Membership
- Order Tracking
- Cancellation
- Damaged/Missing Items
- Gift Cards
- Support Hours

The support assistant follows this flow:

```text
User Question
      ↓
Intent Classification
      ↓
Policy Question? ─────────────── No ─────→ Direct Answer
      │
     Yes
      ↓
Query Embedding
      ↓
ChromaDB Retrieval
      ↓
Top-3 Relevant Policy Chunks
      ↓
Grounded Response
      ↓
Structured JSON Response
```

---

# 2. Module 3 Architecture

The complete architecture is:

```text
                 ┌──────────────────────┐
                 │     User Question    │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │   classify_intent    │
                 └──────────┬───────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ↓                           ↓
     Policy Question              General Question
              │                           │
              ↓                           ↓
     Query Embedding               direct_answer
              │                           │
              ↓                           │
       ChromaDB Search                    │
              │                           │
              ↓                           │
       Top-3 Retrieval                    │
              │                           │
              ↓                           │
   retrieve_and_answer                    │
              │                           │
              └─────────────┬─────────────┘
                            ↓
                  Pydantic Response
                            ↓
                     FastAPI /ask
```

---

# 3. Technologies Used

- Python
- Sentence Transformers
- `all-MiniLM-L6-v2`
- ChromaDB
- LangGraph
- Pydantic
- FastAPI
- Uvicorn
- Docker
- Google Colab for development
- GitHub for project submission

---

# 4. Project Structure

```text
Module-3-GenAI-Support-Assistant/
│
├── chroma_db/
│   └── chroma.sqlite3
│
├── docs/
│   ├── doc_01_delivery.txt
│   ├── doc_02_returns.txt
│   ├── doc_03_membership.txt
│   ├── doc_04_orders.txt
│   ├── doc_05_order_cancel.txt
│   ├── doc_06_damaged_items.txt
│   ├── doc_07_gifts.txt
│   └── doc_08_customer.txt
│
├── src/
├── tests/
├── main.py
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 5. Policy Document Corpus

The project uses exactly 8 policy documents.

| File | Policy Topic |
|---|---|
| `doc_01_delivery.txt` | Delivery |
| `doc_02_returns.txt` | Returns & Refunds |
| `doc_03_membership.txt` | Membership |
| `doc_04_orders.txt` | Order Tracking |
| `doc_05_order_cancel.txt` | Cancellation |
| `doc_06_damaged_items.txt` | Damaged/Missing Items |
| `doc_07_gifts.txt` | Gift Cards |
| `doc_08_customer.txt` | Support Hours |

Each document is loaded from the `docs/` directory.

For the current implementation, each document is represented as one chunk.

---

# 6. Document Ingestion

The documents are loaded from:

```text
docs/
```

The application reads each `.txt` file and creates a document record containing:

- Document ID
- Source
- Text

The resulting document chunks are stored in ChromaDB.

The current project contains:

```text
8 documents
8 chunks
```

---

# 7. Embedding Model

The project uses the open-source Sentence Transformers model:

```text
all-MiniLM-L6-v2
```

The embedding model converts the policy text and user query into numerical vectors.

The generated embedding dimension is:

```text
384
```

Embeddings are normalized before storage and retrieval.

---

# 8. ChromaDB

ChromaDB is used as the vector database.

Persistent database directory:

```text
chroma_db/
```

Collection name:

```text
zepto_policies
```

The collection contains the 8 policy chunks and their embeddings.

The application uses a persistent ChromaDB client so that the vector database can be reused between application runs.

---

# 9. Retrieval

For a policy-related question, the application:

1. Receives the user query.
2. Converts the query into an embedding.
3. Searches the ChromaDB collection.
4. Retrieves the top 3 most similar chunks.
5. Uses the retrieved context to create the response.

The retrieval function is:

```python
retrieve_and_answer()
```

The retrieval flow is:

```text
User Query
    ↓
Query Embedding
    ↓
ChromaDB Similarity Search
    ↓
Top 3 Relevant Chunks
    ↓
Retrieved Context
```

The retrieved chunk IDs are returned in the `sources` field.

---

# 10. Intent Classification

The application uses the `classify_intent` LangGraph node.

The routing is based on a keyword heuristic.

Policy keywords include:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

If a query contains one of these policy keywords, it is classified as:

```text
policy_question
```

Otherwise it is classified as:

```text
general_question
```

The routing does not require an LLM call in the graded baseline.

---

# 11. LangGraph

The application uses LangGraph `StateGraph`.

The graph contains three main nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

### Node 1 — `classify_intent`

Determines whether the query is a policy question or a general question.

### Node 2 — `retrieve_and_answer`

Used for policy questions.

This node:

- Embeds the user query.
- Retrieves the top 3 chunks from ChromaDB.
- Generates the deterministic mock response.
- Returns the retrieved sources.

### Node 3 — `direct_answer`

Used for general questions.

In mock mode it returns:

```text
I can only answer questions about Zepto policies right now.
```

---

# 12. LangGraph Routing

The conditional routing is:

```text
START
  ↓
classify_intent
  │
  ├── policy_question
  │        ↓
  │   retrieve_and_answer
  │        ↓
  │       END
  │
  └── general_question
           ↓
      direct_answer
           ↓
          END
```

---

# 13. Prompt Engineering

The structured prompt template contains the required components:

### Role

Defines the assistant's role as a Zepto customer support assistant.

### Context

Provides the retrieved policy information.

### Task

Instructs the assistant to answer using the retrieved information.

### Format

Specifies the expected structured response.

### Length

Keeps the response concise and relevant.

### Negative Constraint

The prompt explicitly prevents unsupported information from being invented.

Example:

```text
Do not answer using information that is not present in the provided context.
Do not invent policies, prices, timings, refunds, or other details.
```

### Few-Shot Example

The prompt also includes an example showing how a policy question should be answered using retrieved context.

---

# 14. Structured Output

Pydantic is used to validate the response format.

The response model is:

```python
class SupportResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
```

The confidence field is restricted to:

```text
0.0 <= confidence <= 1.0
```

Expected JSON structure:

```json
{
  "answer": "string",
  "sources": ["string"],
  "confidence": 1.0
}
```

For general questions, the `sources` list is empty.

---

# 15. MOCK_LLM Mode

The graded baseline uses:

```text
MOCK_LLM=1
```

This mode is deterministic and does not require an external LLM.

For a policy question, the mock response follows:

```text
Based on the retrieved context: <top retrieved chunk snippet>
```

The retrieved snippet is limited to approximately the first 200 characters.

For a general question, the fixed response is:

```text
I can only answer questions about Zepto policies right now.
```

This allows the complete graded baseline to work without an external LLM API.

---

# 16. Optional Real LLM Path

The project structure allows an optional real LLM extension.

The graded baseline does not require a real LLM or API key.

The default mode remains:

```text
MOCK_LLM=1
```

This keeps the graded implementation deterministic and offline.

---

# 17. LLM Response Validation and Retry

The project includes:

```python
validate_llm_response_with_retry()
```

This helper validates an LLM response against the Pydantic `SupportResponse` schema.

The validation flow is:

```text
Initial LLM Response
        ↓
Pydantic Validation
        │
        ├── Valid
        │     ↓
        │   Return Response
        │
        └── Invalid
              ↓
           Retry 1
              ↓
           Retry 2
              ↓
       Final Error Response
```

The implementation allows up to 2 additional retries.

If all attempts fail, the application returns a clearly marked error response with:

```text
confidence = 0.0
sources = []
```

---

# 18. FastAPI

The application exposes a FastAPI endpoint:

```text
POST /ask
```

Request model:

```json
{
  "query": "string"
}
```

Response model:

```json
{
  "answer": "string",
  "sources": ["string"],
  "confidence": 1.0
}
```

The application also provides:

```text
GET /
```

for a basic health/status response.

---

# 19. Running FastAPI Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

The application will be available at:

```text
http://localhost:7860
```

FastAPI Swagger documentation:

```text
http://localhost:7860/docs
```

---

# 20. API Example — Policy Question

### Request

```json
{
  "query": "How much does delivery cost below INR 149?"
}
```

### Expected response structure

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": [
    "doc_01_delivery_chunk_01",
    "..."
  ],
  "confidence": 1.0
}
```

The exact top-3 source list can depend on the similarity search results.

The important result is that the delivery policy document is retrieved as the relevant source.

---

# 21. API Example — General Question

### Request

```json
{
  "query": "What is the capital of India?"
}
```

### Response

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

This demonstrates the general-question route.

---

# 22. API Validation

FastAPI and Pydantic validate incoming requests.

For example, the expected request contains:

```json
{
  "query": "some question"
}
```

If the required `query` field is missing or invalid, FastAPI returns a validation error such as HTTP `422`.

---

# 23. Docker

The project includes a Dockerfile for local container execution.

### Dockerfile configuration

The application uses:

```text
python:3.11-slim
```

The working directory is:

```text
/app
```

The application exposes:

```text
7860
```

The default environment is:

```text
MOCK_LLM=1
```

The container starts FastAPI using Uvicorn.

---

# 24. Docker Build

From the project directory, run:

```bash
docker build -t zepto-support-assistant .
```

---

# 25. Docker Run

Run the container:

```bash
docker run --name zepto-support -p 7860:7860 zepto-support-assistant
```

Then open:

```text
http://localhost:7860/docs
```

Use the Swagger UI to test:

```text
POST /ask
```

---

# 26. End-to-End Data Flow

The complete system flow is:

```text
8 Zepto Policy Documents
          ↓
       Loading
          ↓
       Chunking
          ↓
Sentence Transformer
all-MiniLM-L6-v2
          ↓
      Embeddings
          ↓
       ChromaDB
  zepto_policies
          ↓
     User Query
          ↓
   Intent Classification
          ↓
   ┌──────┴───────┐
   │              │
Policy          General
Question        Question
   │              │
   ↓              ↓
Embedding      Direct Answer
   │
   ↓
Top-3 Retrieval
   │
   ↓
Retrieved Context
   │
   ↓
Mock/LLM Generation
   │
   ↓
Pydantic Validation
   │
   ↓
Structured Response
   │
   ↓
FastAPI /ask
```

---

# 27. Important Project Functions

| Function | Purpose |
|---|---|
| `load_documents()` | Loads policy documents |
| `initialize_collection()` | Creates/populates ChromaDB |
| `classify_intent()` | Classifies policy/general questions |
| `retrieve_and_answer()` | Retrieves top-3 policy chunks and answers |
| `direct_answer()` | Handles general questions |
| `route_by_intent()` | Controls LangGraph routing |
| `validate_llm_response_with_retry()` | Validates/retries structured LLM responses |
| `build_mock_response()` | Creates final Pydantic response |
| `ask()` | FastAPI `/ask` endpoint |

---

# 28. ChromaDB Collection

Collection:

```text
zepto_policies
```

Stored information includes:

- Chunk ID
- Document ID
- Source
- Document text
- Embedding

The current project contains:

```text
8 policy chunks
```

---

# 29. Testing Summary

The Module 3 implementation was tested for:

- Policy intent classification
- General intent classification
- ChromaDB retrieval
- LangGraph routing
- Deterministic mock response
- Pydantic structured output
- Retry validation
- Final error handling
- FastAPI application import
- FastAPI `/ask` policy request
- FastAPI `/ask` general request
- Invalid request validation

The final Colab audit confirmed:

```text
Completed checks: 9 / 9

MODULE 3 COLAB AUDIT PASSED
```

---

# 30. Requirements

The main Python dependencies are:

```text
fastapi
uvicorn
pydantic
chromadb
sentence-transformers
langgraph
```

Install them using:

```bash
pip install -r requirements.txt
```

---

# 31. Environment Variable

The application uses:

```text
MOCK_LLM
```

Default:

```text
MOCK_LLM=1
```

The graded baseline should use:

```text
MOCK_LLM=1
```

---

# 32. Limitations

The graded baseline is intentionally deterministic.

It:

- Uses the provided Zepto policy corpus.
- Uses keyword-based intent classification.
- Uses ChromaDB for retrieval.
- Uses a deterministic mock generation response.
- Does not require an external LLM API.

The optional real-LLM functionality is not required for the graded baseline.

---

# 33. Final Module 3 Checklist

```text
[x] 8 policy documents
[x] Document ingestion
[x] Document chunking
[x] all-MiniLM-L6-v2 embeddings
[x] ChromaDB persistent storage
[x] Top-3 retrieval
[x] Keyword intent classification
[x] LangGraph StateGraph
[x] classify_intent node
[x] retrieve_and_answer node
[x] direct_answer node
[x] Conditional routing
[x] Structured prompt template
[x] Role component
[x] Context component
[x] Task component
[x] Format component
[x] Length component
[x] Negative constraint
[x] Few-shot example
[x] Pydantic structured response
[x] MOCK_LLM deterministic mode
[x] LLM response validation
[x] Retry logic
[x] FastAPI POST /ask
[x] Policy question example
[x] General question example
[x] Dockerfile
[x] Docker instructions
[x] README
[x] .gitignore
```

---

# 34. Final Project Status

Module 3 — GenAI Support Assistant is implemented as a RAG-based Zepto policy support service.

The system can:

1. Load the 8 Zepto policy documents.
2. Generate embeddings using `all-MiniLM-L6-v2`.
3. Store and query embeddings using ChromaDB.
4. Classify policy and general questions.
5. Route requests using LangGraph.
6. Retrieve the top 3 relevant policy chunks.
7. Generate a deterministic grounded response in mock mode.
8. Validate structured responses using Pydantic.
9. Retry invalid structured responses in the optional LLM path.
10. Serve the assistant through FastAPI.
11. Run locally using Uvicorn.
12. Run as a Docker container.

---

## Module 3 Submission Structure

The final repository should contain:

```text
Capstone-Project/
│
├── Module-1-Data-Pipeline/
│
├── Module-2-Analytics-and-Machine-Learning/
│
└── Module-3-GenAI-Support-Assistant/
    │
    ├── chroma_db/
    │   └── chroma.sqlite3
    │
    ├── docs/
    │   ├── doc_01_delivery.txt
    │   ├── doc_02_returns.txt
    │   ├── doc_03_membership.txt
    │   ├── doc_04_orders.txt
    │   ├── doc_05_order_cancel.txt
    │   ├── doc_06_damaged_items.txt
    │   ├── doc_07_gifts.txt
    │   └── doc_08_customer.txt
    │
    ├── src/
    ├── tests/
    ├── main.py
    ├── Dockerfile
    ├── requirements.txt
    ├── README.md
    └── .gitignore
```

---

## Author

**Sai Mastan Kinthada**

Module 3 — GenAI Support Assistant  
Zepto Data & AI Platform Capstone Project
