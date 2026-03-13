# Legal Contract Analysis — Multi-Agent RAG System

Interactive console application for querying and analyzing legal contracts using a multi-agent RAG architecture.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  OrchestratorAgent                                  │
│  • Classifies intent (ANSWERABLE / OUT_OF_SCOPE)    │
│  • Maintains multi-turn conversation history        │
│  • Routes to specialist agents                      │
└────────────┬──────────────────────────┬─────────────┘
             │                          │
             ▼                          ▼
┌────────────────────┐      ┌──────────────────────────┐
│  RetrievalAgent    │      │  AnswerAgent             │
│  • FAISS search    │─────▶│  • Grounded generation   │
│  • Top-8 chunks    │      │  • Citation enforcement  │
└────────────────────┘      │  • [RISK] flag detection │
                            └──────────────────────────┘
```

**3 agents, clear responsibilities, no over-agentification.**

---

## Setup

```bash
cd Q:\agent
pip install -r requirements.txt

# Configure LLM
cp .env.example .env
# Edit .env → add OPENAI_API_KEY (or configure Groq / Ollama)
```

---

## Run

```bash
# Interactive mode
python main.py

# LLM-as-judge evaluation
python main.py --eval

# Ground-truth labeled evaluation (28 cases, 4 metrics each)
python main.py --labeled
```

---

## Design Decisions

### Chunking — Clause-level splitting
Split on numbered sections (`1.`, `2.`, etc.) matching contract structure.
**Why:** Each clause is a coherent semantic unit. Sliding-window chunking would split obligations mid-clause, degrading retrieval precision.

### Embeddings — `all-MiniLM-L6-v2`
Local sentence-transformers model; no API cost, fast, strong on short passages.
**Trade-off:** Not legal-domain fine-tuned. A model like `legal-bert` would improve precision at the cost of size.

### LLM — GPT-4o-mini / Groq llama-3.1 / Ollama
`temperature=0.1` for near-deterministic answers. `temperature=0` for the router classifier.
Configured via `.env` — swap provider without code changes.

### Retrieval — FAISS + Cross-Encoder Re-ranking
Two-stage pipeline:
1. **FAISS IndexFlatIP** — fast bi-encoder recall of top-20 candidates
2. **CrossEncoder** (`ms-marco-MiniLM-L-6-v2`) — re-scores each (query, chunk) pair together for precise ranking, returns top-8

**Why two stages:** Bi-encoders embed query and chunk independently (fast, ~recall). Cross-encoders see both together (slower, ~precision). Two-stage gives the best of both.
**Trade-off:** Cross-encoder adds latency per query. At scale (>100k chunks), approximate FAISS methods (IVF, HNSW) would also be needed for the recall stage.

### Prompt Design
- **Router prompt:** single-word classification, `temperature=0`, `max_tokens=5` — deterministic routing.
- **Answer prompt:** strict citation format `[DocName §Section]`, `[RISK]` convention, exact value quoting, full document names with abbreviations.

---

## Evaluation

### LLM-as-judge (`--eval`)
Scores each answer on Relevance (0–3) and Groundedness (0–3).
- **Relevance:** catches retrieval failures (wrong chunks returned)
- **Groundedness:** catches hallucination (answer not tied to document text)

### Labeled dataset (`--labeled`)
28 human-labeled cases across 5 categories. 4 deterministic metrics per case:
- **AnswerMatch** — expected keywords present in response
- **CitationMatch** — expected source documents retrieved
- **RiskMatch** — `[RISK]` flag present/absent as expected
- **RefusalMatch** — system refused or answered as expected

**Results: 19/28 (68%) pass rate, 0.84/1.0 avg partial score**

| Category | Cases | Pass |
|---|---|---|
| Factual | 10 | 9 |
| Risk | 7 | 2 |
| CrossDoc | 3 | 1 |
| Hallucination | 4 | 4 |
| OutOfScope | 4 | 3 |

---

## Known Limitations

- Clause-level chunking may miss cross-section context (e.g. definitions in §1 referenced in §4)
- Broad queries (e.g. "summarize all risks") may not retrieve chunks from every document — per-document retrieval would improve coverage
- Conversation history is capped at last 4 messages to control token usage

---

## Sample Queries

```
What is the notice period for terminating the NDA?
What is the uptime commitment in the SLA?
Which law governs the Vendor Services Agreement?
Do confidentiality obligations survive termination of the NDA?
Are there conflicting governing laws across agreements?
Can Vendor XYZ share Acme's confidential data with subcontractors?
Summarize all risks for Acme Corp in one paragraph.
Can you draft a better NDA for me?          ← handled as out-of-scope
```
