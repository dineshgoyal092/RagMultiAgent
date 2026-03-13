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
┌────────────────────────┐   ┌──────────────────────────┐
│  RetrievalAgent        │   │  AnswerAgent             │
│  • FAISS recall (12)   │──▶│  • Grounded generation   │
│  • Cross-encoder rerank│   │  • Citation enforcement  │
│  • Returns top-8       │   │  • [RISK] flag detection │
└────────────────────────┘   └──────────────────────────┘
```

**3 agents, clear responsibilities, no over-agentification.**

---

## Setup

```bash
pip install -r requirements.txt

# Configure LLM provider
cp .env.example .env
# Edit .env — choose one provider:
#   LLM_PROVIDER=groq       → add GROQ_API_KEY
#   LLM_PROVIDER=openai     → add OPENAI_API_KEY
#   LLM_PROVIDER=ollama     → install Ollama and pull a model
```

---

## Run

```bash
# Interactive mode
python main.py

# LLM-as-judge evaluation (17 queries, relevance + groundedness)
python main.py --eval

# Ground-truth labeled evaluation (28 cases, 4 metrics each)
python main.py --labeled
```

---

## LLM Providers

The system supports three providers — switch by changing `LLM_PROVIDER` in `.env`, no code changes needed.

| Provider | Config | Notes |
|---|---|---|
| Groq | `LLM_PROVIDER=groq` | Fast, free tier, cloud |
| OpenAI | `LLM_PROVIDER=openai` | GPT-4o-mini default |
| Ollama | `LLM_PROVIDER=ollama` | Local, no API cost |

**Ollama setup:**
```bash
# Install from https://ollama.com, then:
ollama pull llama3
# Set LLM_PROVIDER=ollama in .env
```

---

## Design Decisions

### Chunking — Clause-level splitting
Split on numbered sections (`1.`, `2.`, etc.) matching contract structure.
**Why:** Each clause is a coherent semantic unit. Sliding-window chunking would split obligations mid-clause, degrading retrieval precision.

### Embeddings — `all-MiniLM-L6-v2`
Local sentence-transformers model; no API cost, fast, strong on short passages.
**Trade-off:** Not legal-domain fine-tuned. A model like `legal-bert` would improve precision at the cost of size.

### LLM — Configurable (Groq / OpenAI / Ollama)
`temperature=0.1` for near-deterministic answers. `temperature=0` for the router classifier.
Configured via `.env` — swap provider without code changes.

### Retrieval — FAISS + Cross-Encoder Re-ranking
Two-stage pipeline:
1. **FAISS IndexFlatIP** — fast bi-encoder recall of top-12 candidates (exact cosine similarity)
2. **CrossEncoder** (`ms-marco-MiniLM-L-6-v2`) — re-scores each (query, chunk) pair together for precise ranking, returns top-8

**Why two stages:** Bi-encoders embed query and chunk independently (fast, ~recall). Cross-encoders see both together (slower, ~precision). Two-stage gives the best of both.
**Trade-off:** Cross-encoder adds latency per query. At scale (>100k chunks), approximate FAISS methods (IVF, HNSW) would also be needed for the recall stage.

### Prompt Design
- **Router prompt:** single-word classification (`ANSWERABLE` / `OUT_OF_SCOPE`), `temperature=0`, `max_tokens=5` — cheap and deterministic.
- **Answer prompt:** strict citation format `[DocName §Section]`, `[RISK]` prefix convention, verbatim value quoting, full document names with abbreviations.

---

## Evaluation

### LLM-as-judge (`--eval`)
Scores each of the 17 assignment queries on two axes using an LLM judge:
- **Relevance (0–3):** did the answer address the question?
- **Groundedness (0–3):** is the answer tied to document text with citations?

**Results: Avg Relevance 2.9/3 | Avg Groundedness 2.2/3**

**Why these metrics:**
- Relevance catches routing and retrieval failures
- Groundedness catches hallucination
- **Limitation:** judge model can be inconsistent across runs; no absolute ground truth

### Labeled dataset (`--labeled`)
28 human-labeled cases across 5 categories with 4 deterministic metrics per case:
- **AnswerMatch** — expected keywords present in response
- **CitationMatch** — expected source documents retrieved
- **RiskMatch** — `[RISK]` flag present/absent as expected
- **RefusalMatch** — system refused or answered as expected

| Category | Cases | Pass |
|---|---|---|
| Factual | 10 | 9 |
| Risk | 7 | 2 |
| CrossDoc | 3 | 1 |
| Hallucination | 4 | 4 |
| OutOfScope | 4 | 3 |
| **Total** | **28** | **19 (68%)** |

---

## Known Limitations

- Clause-level chunking may miss cross-section context (e.g. definitions in §1 referenced in §4)
- Compound queries (e.g. "What is X and what is Y?") may miss one part — single embedding averages both intents; query decomposition would fix this
- Conversation history is capped at last 4 messages to control token usage
- Cross-encoder adds per-query latency; not suitable for high-throughput production without batching

---

## Sample Queries

```
What is the notice period for terminating the NDA?
What is the uptime commitment in the SLA?
Which law governs the Vendor Services Agreement?
Do confidentiality obligations survive termination of the NDA?
Is liability capped for breach of confidentiality?
Are there conflicting governing laws across agreements?
Can Vendor XYZ share Acme's confidential data with subcontractors?
Summarize all risks for Acme Corp in one paragraph.
Can you draft a better NDA for me?                    ← out-of-scope refusal
What legal strategy should Acme take against Vendor XYZ?  ← out-of-scope refusal
```
