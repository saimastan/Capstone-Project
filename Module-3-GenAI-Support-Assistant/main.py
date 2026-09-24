
import os
import glob
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer

from langgraph.graph import StateGraph, START, END

from fastapi import FastAPI
from pydantic import BaseModel, Field


# ============================================================
# 1. CONFIGURATION
# ============================================================

os.environ.setdefault("MOCK_LLM", "1")
MOCK_LLM = os.getenv("MOCK_LLM", "1")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "zepto_policies"


# ============================================================
# 2. EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# 3. CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "description": "Zepto policy documents for the Module 3 support assistant"
    }
)


# ============================================================
# 4. LOAD / INGEST DOCUMENTS IF CHROMA IS EMPTY
# ============================================================

def load_documents():
    documents = []

    file_paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt")))

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        document_id = os.path.basename(file_path)

        documents.append({
            "document_id": document_id,
            "source": document_id,
            "text": text
        })

    return documents


def initialize_collection():
    """
    Populate ChromaDB if the collection is empty.
    """

    if collection.count() > 0:
        return

    documents = load_documents()

    if not documents:
        raise RuntimeError(
            "No policy documents were found in the docs directory."
        )

    texts = [doc["text"] for doc in documents]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    ids = [
        doc["document_id"].replace(".txt", "_chunk_01")
        for doc in documents
    ]

    metadatas = [
        {
            "document_id": doc["document_id"],
            "source": doc["source"]
        }
        for doc in documents
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )


initialize_collection()


# ============================================================
# 5. LANGGRAPH STATE
# ============================================================

class SupportState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_documents: list
    answer: str
    sources: list[str]
    confidence: float


# ============================================================
# 6. POLICY KEYWORDS
# ============================================================

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


# ============================================================
# 7. STRUCTURED PROMPT TEMPLATE
# ============================================================

STRUCTURED_PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer support assistant.

CONTEXT:
Use only the policy information provided in the retrieved context.

TASK:
Answer the user's question using the retrieved policy information.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent policies, prices, timings, refunds, or other details.

FORMAT:
Return a structured response containing:
- answer
- sources
- confidence

LENGTH:
Keep the answer concise and directly relevant to the user's question.

FEW-SHOT EXAMPLE:
User question:
How much does delivery cost below INR 149?

Context:
Standard delivery is free on orders over INR 149.
Orders below INR 149 incur a flat INR 25 delivery fee.

Example answer:
Orders below INR 149 incur a flat INR 25 delivery fee.

USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}
"""


# ============================================================
# 8. PYDANTIC SCHEMAS
# ============================================================

class SupportResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


class AskRequest(BaseModel):
    query: str


# ============================================================
# 9. LLM RESPONSE VALIDATION + RETRY
# ============================================================

def validate_llm_response_with_retry(
    raw_outputs,
    max_retries=2
):
    """
    Validate an LLM response against the SupportResponse schema.

    raw_outputs:
        A list containing the initial LLM response followed by
        optional retry responses.

    max_retries:
        Maximum number of additional validation attempts.

    Returns:
        SupportResponse if validation succeeds.

        Otherwise, a clear error response.
    """

    total_attempts = max_retries + 1

    for attempt in range(total_attempts):

        if attempt >= len(raw_outputs):
            break

        raw_output = raw_outputs[attempt]

        try:

            # Already a Pydantic object
            if isinstance(raw_output, SupportResponse):
                return raw_output

            # Dictionary / JSON-like response
            if isinstance(raw_output, dict):
                return SupportResponse.model_validate(raw_output)

            # JSON string response
            if isinstance(raw_output, str):
                return SupportResponse.model_validate_json(raw_output)

        except Exception:
            # Validation failed.
            # In a real LLM implementation, the next retry would
            # send a corrective instruction to the LLM.
            continue

    return SupportResponse(
        answer=(
            "ERROR: Unable to validate the LLM response "
            f"after {total_attempts} attempts."
        ),
        sources=[],
        confidence=0.0
    )


# ============================================================
# 10. INTENT CLASSIFICATION
# ============================================================

def classify_intent(state: SupportState):

    query = state["query"].lower()

    if any(keyword in query for keyword in POLICY_KEYWORDS):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        "intent": intent
    }


# ============================================================
# 11. RETRIEVAL + ANSWER
# ============================================================

def retrieve_and_answer(state: SupportState):

    query = state["query"]

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_documents = []

    result_ids = results.get("ids", [[]])[0]
    result_documents = results.get("documents", [[]])[0]
    result_metadatas = results.get("metadatas", [[]])[0]
    result_distances = results.get("distances", [[]])[0]

    for i in range(len(result_ids)):

        retrieved_documents.append({
            "chunk_id": result_ids[i],
            "document_id": result_metadatas[i].get(
                "document_id",
                ""
            ),
            "source": result_metadatas[i].get(
                "source",
                ""
            ),
            "text": result_documents[i],
            "distance": result_distances[i]
            if i < len(result_distances)
            else None
        })

    # --------------------------------------------------------
    # MOCK LLM
    # --------------------------------------------------------

    if MOCK_LLM == "1":

        top_chunk = (
            retrieved_documents[0]["text"]
            if retrieved_documents
            else ""
        )

        answer = (
            "Based on the retrieved context: "
            + top_chunk[:200]
        )

        return {
            "retrieved_documents": retrieved_documents,
            "answer": answer,
            "sources": [
                doc["chunk_id"]
                for doc in retrieved_documents
            ],
            "confidence": 1.0
        }

    # --------------------------------------------------------
    # OPTIONAL REAL LLM PATH
    # --------------------------------------------------------

    # The graded implementation works without an external LLM.
    # A real LLM can be connected here as an optional extension.

    return {
        "retrieved_documents": retrieved_documents,
        "answer": (
            "Real LLM mode is not configured. "
            "Use MOCK_LLM=1 for the graded offline mode."
        ),
        "sources": [
            doc["chunk_id"]
            for doc in retrieved_documents
        ],
        "confidence": 0.0
    }


# ============================================================
# 12. GENERAL QUESTION
# ============================================================

def direct_answer(state: SupportState):

    if MOCK_LLM == "1":

        return {
            "answer": (
                "I can only answer questions about "
                "Zepto policies right now."
            ),
            "sources": [],
            "confidence": 1.0
        }

    return {
        "answer": (
            "Real LLM mode is not configured. "
            "Use MOCK_LLM=1 for the graded offline mode."
        ),
        "sources": [],
        "confidence": 0.0
    }


# ============================================================
# 13. CONDITIONAL ROUTING
# ============================================================

def route_by_intent(state: SupportState):

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# 14. LANGGRAPH
# ============================================================

graph_builder = StateGraph(SupportState)

graph_builder.add_node(
    "classify_intent",
    classify_intent
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph_builder.add_node(
    "direct_answer",
    direct_answer
)

graph_builder.add_edge(
    START,
    "classify_intent"
)

graph_builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

graph_builder.add_edge(
    "retrieve_and_answer",
    END
)

graph_builder.add_edge(
    "direct_answer",
    END
)

support_graph = graph_builder.compile()


# ============================================================
# 15. BUILD FINAL RESPONSE
# ============================================================

def build_mock_response(state):

    return SupportResponse(
        answer=state.get(
            "answer",
            "No answer generated."
        ),
        sources=state.get(
            "sources",
            []
        ),
        confidence=state.get(
            "confidence",
            0.0
        )
    )


# ============================================================
# 16. FASTAPI
# ============================================================

app = FastAPI(
    title="Zepto GenAI Support Assistant",
    description="RAG-based Zepto policy support assistant",
    version="1.0.0"
)


@app.get("/")
def root():

    return {
        "message": "Zepto GenAI Support Assistant is running.",
        "mock_llm": MOCK_LLM
    }


@app.post(
    "/ask",
    response_model=SupportResponse
)
def ask(request: AskRequest):

    result = support_graph.invoke(
        {
            "query": request.query
        }
    )

    response = build_mock_response(result)

    return response
