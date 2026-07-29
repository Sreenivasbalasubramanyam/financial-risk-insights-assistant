# Financial Risk Insights Assistant

A retrieval-augmented generation (RAG) pipeline for financial risk, fraud, and
compliance knowledge lookup. Analysts ask natural-language questions —
*"What is the policy on counterparty exposure limits?"* — and get back a
concise, **citation-grounded** answer sourced directly from internal policy
documents, transaction narratives, and analyst notes, exposed through a
FastAPI service.

This project was built to run **fully offline by default** — no OpenAI API
key, no external network calls, no GPU. It's designed so a recruiter (or
anyone) can clone it, run three commands, and get a working demo in under a
minute, while also showing the pluggable seams needed to swap in OpenAI,
sentence-transformer embeddings, FAISS, or LangChain in a real deployment.

## Problem Statement

Risk and fraud analysts spend significant time manually searching across
scattered policy PDFs, compliance memos, and case notes to answer routine
questions ("What's our SAR filing deadline?", "What triggers a fraud
escalation?"). This project demonstrates how a lightweight RAG system can
make that internal knowledge instantly searchable and explainable — every
answer is traceable back to the exact source document and chunk it came
from, which matters a great deal in a regulated, audit-sensitive domain.

## Architecture

```
                        ┌─────────────────────────────┐
                        │   data/policy_documents/     │
                        │   (5 synthetic .txt sources)  │
                        └───────────────┬──────────────┘
                                        │
                                        ▼
                        ┌─────────────────────────────┐
                        │        src/ingest.py          │
                        │  1. load_documents()          │
                        │  2. chunk_document()  ────────┼──> src/chunking.py
                        │     (recursive char splitter,  │    (paragraph > line >
                        │      chunk_size=700, overlap=120)│   sentence > word split,
                        │  3. embedder.fit_transform()  ─┼──> src/embeddings.py
                        │     (TF-IDF, numpy, no deps)   │    (TfidfEmbedder /
                        │  4. index.add(vectors)  ───────┼──> SentenceTransformerEmbedder)
                        │     (cosine index)             │   src/vector_index.py
                        │  5. persist to data/index/     │   (NumpyCosineIndex / FaissIndex)
                        └───────────────┬──────────────┘
                                        │  data/index/{vectors.npz, vocabulary.json,
                                        │              chunks_metadata.json}
                                        ▼
   question ──────────▶  ┌─────────────────────────────┐
                        │       src/retriever.py         │
                        │  embed query -> similarity     │
                        │  search -> top-k chunks with   │
                        │  {source, chunk_id, score}     │
                        └───────────────┬──────────────┘
                                        ▼
                        ┌─────────────────────────────┐
                        │      src/llm_backend.py        │
                        │  ExtractiveAnswerBackend        │  <- default, no API key
                        │    (quotes ranked sentences,    │
                        │     inline [source#chunk] tags) │
                        │  OpenAIAnswerBackend             │  <- opt-in, needs OPENAI_API_KEY
                        │    (Chat Completions + prompt    │
                        │     template, same citations)    │
                        └───────────────┬──────────────┘
                                        ▼
                        ┌─────────────────────────────┐
                        │     src/rag_pipeline.py         │
                        │  answer_query(question) -> dict │
                        │  {answer, backend, citations}    │
                        └───────────────┬──────────────┘
                                        ▼
                        ┌─────────────────────────────┐
                        │         src/api.py              │
                        │  FastAPI: POST /query           │
                        │           GET  /health           │
                        └─────────────────────────────┘
```

## Tech Stack

| Layer                | Default (offline)                          | Pluggable upgrade                              |
|-----------------------|---------------------------------------------|--------------------------------------------------|
| Document chunking      | Hand-rolled recursive char splitter (`src/chunking.py`) | LangChain `RecursiveCharacterTextSplitter` (`src/langchain_variant.py`) |
| Embeddings             | From-scratch TF-IDF + numpy (`src/embeddings.py::TfidfEmbedder`) | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector index           | In-memory numpy cosine similarity (`src/vector_index.py::NumpyCosineIndex`) | `faiss-cpu` (`IndexFlatIP`) |
| Answer generation      | Extractive, citation-quoting (`src/llm_backend.py::ExtractiveAnswerBackend`) | OpenAI Chat Completions (`OpenAIAnswerBackend`) / LangChain `ChatOpenAI` chain |
| Prompt templating      | Plain f-string template (`PROMPT_TEMPLATE`) | LangChain `PromptTemplate` (shown in `src/langchain_variant.py`) |
| API                    | FastAPI + Uvicorn                            | — |
| Testing                | pytest                                       | — |

### Why these defaults?

This project was built and verified in a sandboxed environment with **no
outbound network access**, which is a realistic stand-in for a
security-locked-down analyst workstation or an air-gapped internal tool. So
every layer needed a dependency-light default that still produces a
genuinely useful, correct result:

- **TF-IDF instead of sentence-transformers by default**: `TfidfEmbedder` is
  ~130 lines of plain Python + numpy (tokenize, term/inverse-document
  frequency, L2-normalize). It has zero model-download requirement and is
  fast enough for the corpus sizes this kind of tool typically deals with.
  If `sentence-transformers` is installed, set
  `EMBEDDING_BACKEND=sentence-transformers` to get dense semantic embeddings
  with the exact same downstream interface — no other code changes needed.
- **In-memory numpy cosine index instead of FAISS by default**: for a
  document collection in the hundreds-to-low-thousands of chunks range (this
  demo indexes 26 chunks from 5 documents), a plain `vectors @ query` matrix
  multiply is both simpler to read and fast enough — no compiled dependency,
  no version-fragile wheel. Set `VECTOR_INDEX_BACKEND=faiss` to swap in
  `faiss.IndexFlatIP` when `faiss-cpu` is available.
- **Extractive answer backend instead of an LLM by default**: rather than
  calling an LLM (and needing an API key), `ExtractiveAnswerBackend` selects
  the most lexically relevant *verbatim* sentence from each retrieved chunk
  and presents it with an inline `[source_file.txt#chunk_N]` citation. This
  has a genuinely useful property for a risk/compliance tool: **it cannot
  hallucinate a fact that isn't in the source documents**, because every
  word in the answer is a direct quote. Set `LLM_BACKEND=openai` and export
  `OPENAI_API_KEY` to instead get a fluent, LLM-synthesized answer from the
  same retrieved context and the same citation-grounded prompt template.

## Project Layout

```
financial-risk-insights-assistant/
├── data/
│   └── policy_documents/       # 5 synthetic .txt sources (fictional bank)
├── src/
│   ├── config.py                # env-var-driven settings
│   ├── chunking.py               # recursive character text splitter
│   ├── embeddings.py             # TfidfEmbedder + SentenceTransformerEmbedder
│   ├── vector_index.py           # NumpyCosineIndex + FaissIndex
│   ├── ingest.py                 # load -> chunk -> embed -> index -> persist
│   ├── retriever.py              # load index, similarity search, top-k + metadata
│   ├── llm_backend.py            # ExtractiveAnswerBackend + OpenAIAnswerBackend
│   ├── rag_pipeline.py           # answer_query(question) -> {answer, citations}
│   ├── api.py                    # FastAPI app: /query, /health
│   └── langchain_variant.py      # reference-only LangChain drop-in (commented)
├── tests/
│   ├── test_retriever.py         # tiny in-memory index, checks top-1 relevance
│   └── test_rag_pipeline.py      # end-to-end answer_query() against real data/
├── requirements.txt
├── Dockerfile
├── LICENSE (MIT)
└── README.md
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the vector index from data/policy_documents/
python -m src.ingest

# 3. Run the test suite
python -m pytest tests/ -v

# 4. Ask a question from the CLI
python -m src.rag_pipeline "What is the policy on counterparty exposure limits?"

# 5. Or start the API
uvicorn src.api:app --reload --port 8000
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the policy on counterparty exposure limits?"}'
```

### Docker

```bash
docker build -t financial-risk-insights-assistant .
docker run -p 8000:8000 financial-risk-insights-assistant
```

### Switching backends

```bash
# Use OpenAI for answer generation (requires an API key; falls back to the
# extractive backend automatically if the key is missing or openai isn't installed)
export OPENAI_API_KEY=sk-...
export LLM_BACKEND=openai
python -m src.rag_pipeline "How are fraudulent transactions detected and scored?"

# Use dense sentence-transformer embeddings + FAISS instead of TF-IDF + numpy
export EMBEDDING_BACKEND=sentence-transformers
export VECTOR_INDEX_BACKEND=faiss
python -m src.ingest   # rebuild the index with the new backend
```

## Example Query + Answer

**Request:**
```json
POST /query
{"question": "What is the policy on counterparty exposure limits?"}
```

**Response (default extractive backend, verified output):**
```json
{
  "question": "What is the policy on counterparty exposure limits?",
  "answer": "Based on the retrieved policy and analyst documents, here is what is relevant to: \"What is the policy on counterparty exposure limits?\"\n1. No single counterparty rated investment-grade (BBB- or above) may account for more than 8% of the Bank's Tier 1 capital in aggregate exposure. [risk_policy_counterparty_exposure.txt#chunk_1] (relevance=0.328)\n2. This policy establishes the framework for measuring, monitoring, and limiting counterparty credit exposure across all trading and lending desks at Meridian Fictional Bank (\"the Bank\"). [risk_policy_counterparty_exposure.txt#chunk_0] (relevance=0.286)\n3. Repeated settlement fails with a single counterparty (three or more in a rolling 90-day period) should be factored into the counterparty's internal risk rating and considered during the quarterly exposure limit review described in the Counterparty Exposure Risk Policy. [analyst_notes_settlement_risk.txt#chunk_4] (relevance=0.268)",
  "backend": "extractive",
  "citations": [
    {"source": "risk_policy_counterparty_exposure.txt", "chunk_id": "risk_policy_counterparty_exposure.txt::chunk_1", "chunk_index": 1, "score": 0.3282},
    {"source": "risk_policy_counterparty_exposure.txt", "chunk_id": "risk_policy_counterparty_exposure.txt::chunk_0", "chunk_index": 0, "score": 0.2863},
    {"source": "analyst_notes_settlement_risk.txt", "chunk_id": "analyst_notes_settlement_risk.txt::chunk_4", "chunk_index": 4, "score": 0.2676}
  ]
}
```

Every claim in the answer is traceable to a specific document and chunk —
useful for compliance sign-off and for analysts who need to pull up the
source document to verify context.

## Testing

- `tests/test_retriever.py` builds a tiny 3-chunk in-memory index and
  asserts that a counterparty-exposure question ranks the counterparty-
  exposure chunk first (and similarly for a fraud-detection question),
  verifying the retrieval math in isolation.
- `tests/test_rag_pipeline.py` runs `answer_query()` end-to-end against the
  real `data/policy_documents/` corpus and asserts the answer is non-empty,
  uses the extractive backend by default, and contains an inline citation
  tag pointing at the expected source document.

```bash
python -m pytest tests/ -v
```

## Future Improvements

- Add hybrid retrieval (TF-IDF + dense embeddings, rank-fused) for better
  recall on paraphrased questions.
- Add metadata filtering (e.g. filter to `policy` vs `analyst_note` vs
  `transaction_narrative` document types) as a query parameter.
- Add a re-ranking stage (cross-encoder) before answer generation once a
  larger corpus makes single-stage retrieval less precise.
- Add conversation memory / follow-up question support in the API.
- Add streaming responses for the OpenAI backend.
- Add authentication/rate-limiting to `src/api.py` before any real internal
  deployment.
- Expand the synthetic corpus and add a small labeled eval set (question →
  expected source document) to track retrieval quality over time.

## License

MIT License. See [LICENSE](LICENSE). Copyright (c) 2026 Sreenivas Balasubramanyam.

All data in `data/policy_documents/` is entirely synthetic and fictional,
created for demonstration purposes only. It does not represent any real
financial institution, policy, customer, or transaction.
