
# Zepto GenAI Support Assistant

## 1. Project Overview

This project implements a GenAI-powered support assistant for Zepto policy questions.

The assistant uses:

- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- ChromaDB
- LangGraph
- Pydantic
- FastAPI
- Docker
- Deterministic offline mock LLM mode

The graded version runs completely offline using:

```text
MOCK_LLM=1
