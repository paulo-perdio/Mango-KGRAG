<img width="1556" height="784" alt="preview (1)" src="https://github.com/user-attachments/assets/ce40b748-e6e6-4cba-ab1a-97240ed52cf6" /># Mango AI — Knowledge Graph-Driven RAG for Local Thai Plant Nomenclature

A small FastAPI chatbot that answers Thai-language questions about mango cultivation
by combining a knowledge-graph query-expansion step with retrieval-augmented
generation (KG-RAG). Built from the engineering behind my KMITL capstone project,
*"Knowledge Graph-Driven Prompt Augmentation for RAG Systems: A Case Study of Local
Thai Herb Nomenclature"* (Anon Thongsawaeng, John Paul L. Perdio, Phalat Kraichoke;
advisor: Assoc. Prof. Dr. Rathachai Chawuthai, KMITL, 2025–2026).

## What's real vs. what's a stand-in

The capstone's actual knowledge graph (the CAVOC ontology, provided by our advisor)
and its 100-question benchmark are private research data and aren't included here.
**The data in `/data` in this repo is a small dataset I wrote myself**, in the same
format the pipeline expects, so the app is fully runnable end-to-end by anyone who
clones it. It is not the original research corpus, and the numbers you get from
running it are not the paper's reported results.

For the real, published findings — a 200x average retrieval-rank improvement from
KG-based query expansion, and a 16% increase in answerable questions from reranking —
see the capstone report.

## Architecture

```
User query (Thai)
      │
      ▼
Chit-chat filter ──► if greeting/small talk ──► Llama-3.2-1B direct answer
      │
      ▼ (real question)
Ontology + RAG relevance scoring (BGE-m3 embeddings)
      │
      ├─ below threshold ──► Llama-3.2-1B direct answer (no retrieval)
      │
      ▼ above threshold
Hybrid triple retrieval (lexical filter + dense re-score) from the KG
      │
      ▼
Query expansion: original query + verbalized triples
      │
      ▼
RAG chunk retrieval (cosine similarity) + cross-encoder reranking
      │
      ▼
Llama-3.2-1B generates the final answer from triples + chunks + query
```

Models used: `meta-llama/Llama-3.2-1B-Instruct` (generation), `BAAI/bge-m3`
(embeddings), `cross-encoder/ms-marco-MiniLM-L-12-v2` (reranking).

## Setup

**Requirements:** Python 3.10+, ~6GB free disk for model downloads, and a
HuggingFace account with access to Llama-3.2-1B-Instruct (accept the license on
the model page first). A GPU is strongly recommended — this will run on CPU but
slowly.

```bash
git clone <your-repo-url>
cd mango-kgrag
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste in your own HF_TOKEN

uvicorn app:app --reload
```

Then open http://127.0.0.1:8000 in your browser.

First run will download the three models (a few GB total) and compute embedding
caches for the sample data — this takes a few minutes. Subsequent runs load from
cache and start in seconds.

## Try asking

- "มะม่วงน้ำดอกไม้เก็บเกี่ยวช่วงไหน" (when is Nam Dok Mai mango harvested)
- "โรคแอนแทรคโนสคืออะไร" (what is anthracnose)
- "สวัสดี" (hello — routes to the chit-chat path, no retrieval)

The debug panel on the right shows which mode was used (`kgrag` vs `llama_only`),
the retrieved ontology triples, and the RAG chunks that fed the answer.

<img width="1556" height="784" alt="preview (1)" src="https://github.com/user-attachments/assets/3d58b5d5-788f-4acf-aae4-308e6a1e0d2a" />
<img width="1568" height="709" alt="preview (2)" src="https://github.com/user-attachments/assets/8760ace8-c677-4294-ba34-5738a7bfeed7" />
<img width="1548" height="784" alt="preview" src="https://github.com/user-attachments/assets/30b13696-9d53-4065-b12f-80c537023bcb" />

## Known Limitations

- The sample ontology (20 triples) and RAG corpus (14 passages) are intentionally
  small — enough to demonstrate the pipeline working end-to-end, not to reproduce
  the paper's retrieval-quality numbers.
- Thai→English translation (NLLB-200) is loaded but bypassed in the active answer
  path, matching the original research code's final configuration.
- No automatic GPU fallback tuning — if you're CPU-only, expect each answer to
  take significantly longer.
- **The relevance gate can false-positive on off-topic questions that share
  vocabulary with the ontology.** For example, asking about durian harvest
  timing can incorrectly trigger KG-RAG mode, because the ontology predicate
  `harvest_season` shares the word "harvest" with the query. On a small,
  single-domain sample corpus like this one, a single shared word is a larger
  fraction of the similarity signal than it would be against a large, diverse
  corpus (as in the real research dataset). When this happens, the model
  generates an answer that is not actually grounded in the retrieved data and
  can include fabricated details. This is a real limitation of max-cosine-
  similarity gating at this corpus scale, not a bug I've silently patched
  over — a production system would need a more robust relevance signal (e.g.
  a trained classifier, or an entity/topic check) rather than a single
  similarity threshold.
- Generation quality on Thai output is uneven at the token level (occasional
  stray characters, occasional date imprecision even when the correct fact was
  retrieved) — expected behavior for a 1B-parameter model, not something tuned
  away by adjusting `repetition_penalty` alone (see commit history for what
  was tried).
