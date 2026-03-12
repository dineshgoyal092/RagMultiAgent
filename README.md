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
│  • Top-4 chunks    │      │  • Citation enforcement  │
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
# Edit .env → add OPENAI_API_KEY (or configure Ollama)
```

---

## Run

```bash
# Interactive mode
python main.py

# Evaluation suite
python main.py --eval
```

---

## Design Decisions

### Chunking — Clause-level splitting
Split on numbered sections (`1.`, `2.`, etc.) matching contract structure.
**Why:** Each clause is a coherent semantic unit. Sliding-window chunking would split obligations mid-clause, degrading retrieval precision.

### Embeddings — `all-MiniLM-L6-v2`
Local sentence-transformers model; no API cost, fast, strong on short passages.
**Trade-off:** Not legal-domain fine-tuned. A model like `legal-bert` would improve precision at the cost of size.

### LLM — GPT-4o-mini (or Ollama llama3)
`temperature=0.1` for near-deterministic answers. `temperature=0` for the router classifier.
Configured via `.env` — swap provider without code changes.

### Retrieval — FAISS IndexFlatIP
Exact cosine similarity over normalized embeddings. Correct at this corpus size (~20 chunks).
**Trade-off:** At scale (>100k chunks), approximate methods (IVF, HNSW) would be needed.

### Prompt Design
- **Router prompt:** single-word classification, `temperature=0`, `max_tokens=5` — deterministic routing.
- **Answer prompt:** strict citation format `[DocName §Section]`, `[RISK]` convention, no legal opinions beyond documents.

---

## Evaluation

**What:** LLM-as-judge scoring each answer on Relevance (0–3) and Groundedness (0–3).
**Why Relevance:** catches retrieval failures (wrong chunks returned).
**Why Groundedness:** catches hallucination (answer not tied to document text).
**Limitations:** no ground-truth labels; judge model can be inconsistent; small corpus limits coverage testing.

---

## Known Limitations

- No re-ranking (cross-encoder reranking would improve precision on ambiguous queries)
- Clause-level chunking may miss cross-section context (e.g. definitions in §1 referenced in §4)
- Evaluation is relative, not absolute — no human-labeled gold answers
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
