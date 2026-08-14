# Mango AI — Knowledge Graph-Driven RAG for Local Thai Plant Nomenclature

A small FastAPI chatbot that answers Thai-language questions about mango cultivation
by combining a knowledge-graph query-expansion step with retrieval-augmented
generation (KG-RAG). Built from the engineering behind my KMITL capstone project, *"Knowledge Graph-Driven Prompt Augmentation for RAG Systems: A Case Study of Local
Thai Herb Nomenclature"* (Anon Thongsawaeng, John Paul L. Perdio, Phalat Kraichoke;
advisor: Assoc. Prof. Dr. Rathachai Chawuthai, KMITL, 2025–2026).

## What's real vs. what's a stand-in

The capstone's actual knowledge graph (the CAVOC ontology, provided by our advisor)
and its 100-question benchmark are private research data and aren't included here.
The data in `/data` in this repo is a small dataset I wrote myself, in the same
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
Ontology + RAG relevance scoring (BGE-m3 embeddings, computed once per query)
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

Models used: `meta-llama/Llama-3.2-1B-Instruct` (generation), `BAAI/bge-m3` (embeddings, loaded once and shared across ontology and RAG retrieval), `cross-encoder/ms-marco-MiniLM-L-12-v2` (reranking).

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

Then open <http://127.0.0.1:8000> in your browser.

First run will download the two models (a few GB total) and compute embedding
caches for the sample data — this takes a few minutes. Subsequent runs load from
cache and start in seconds.

## Try asking

- "มะม่วงน้ำดอกไม้เก็บเกี่ยวช่วงไหน" (when is Nam Dok Mai mango harvested)
- "โรคแอนแทรคโนสคืออะไร" (what is anthracnose)
- "สวัสดี" (hello — routes to the chit-chat path, no retrieval)

The debug panel on the right shows which mode was used (`kgrag` vs `llama_only`),
the retrieved ontology triples, and the RAG chunks that fed the answer.

## Engineering cleanup (post-capstone)

The version of this code that came directly out of the capstone worked, but had
several things a from-scratch code review would flag. Fixed here, documented so
the change is visible rather than silent:

- **Duplicate embedding model.** BGE-m3 was being loaded twice under two
  different variable names (once for ontology triples, once for RAG chunks) —
  same weights, twice the memory and load time for no reason. Now loaded once
  and shared.
- **Dead translation pipeline.** The original research code translated Thai
  queries to English before retrieval; this deployed version never actually
  calls that step (`query_en = query`, no-op). The NLLB-200 model (~600MB) was
  still being loaded on every startup despite never being used. Removed —
  translation code and model load are gone rather than sitting dead in the
  file. If real translation is reinstated later, it'll need to be added back
  deliberately, not silently re-enabled.
- **Redundant query embeddings.** A single incoming query was being embedded
  up to three separate times per request (once each for the ontology relevance
  check, the RAG relevance check, and hybrid triple retrieval) with no reuse.
  Now embedded once per request and passed through.
- **Debug print left in the generation path.** Removed.
- **Chit-chat filter was too aggressive.** Any query ≤4 characters was routed
  away from retrieval entirely, which risked misclassifying short but real
  Thai questions. Narrowed to ≤2 characters. This is a heuristic, not a tuned
  threshold — it hasn't been validated against a real query distribution.
- **Hardcoded retrieval mode.** `MODE = "hybrid"` was set inline inside the
  answer function. Pulled out to a module-level `RETRIEVAL_MODE` env var
  (`RETRIEVAL_MODE=dense` / `hybrid` / `cross`), default unchanged.
- Removed ~10 unused imports (`rdflib`, `pandas`, `time`, `re`, `pythainlp`,
  `nltk.ngrams`, `BertTokenizer`/`BertForMaskedLM`/`BertModel`, `bert_score`,
  `rouge`, `rouge_score`) left over from the research notebook's evaluation
  code — none were referenced anywhere in the deployed app.

## Known limitations of this demo

- The sample ontology (20 triples) and RAG corpus (14 passages) are intentionally
  small — enough to demonstrate the pipeline working end-to-end, not to reproduce
  the paper's retrieval-quality numbers.
- No Thai→English translation happens in this deployed version — see "Engineering
  cleanup" above. The pipeline runs on the raw query text directly.
- The chit-chat filter (≤2 characters, or an exact match against a small greeting
  list) is a blunt heuristic, not a validated classifier.
- No automatic GPU fallback tuning — if you're CPU-only, expect each answer to
  take significantly longer.

## About

No description, website, or topics provided.