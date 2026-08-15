import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel
import numpy as np
import torch.nn.functional as F
import json
from sklearn.metrics.pairwise import cosine_similarity
import os
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------
# CONFIGURATION
# ----------------------------
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads a local .env file if present (never commit this file)

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
DATA_DIR = BASE_DIR / "data"

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
EMBED_MODEL = "BAAI/bge-m3"
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is not set. Create a .env file (see .env.example) or "
        "export HF_TOKEN=your_token before running."
    )

DEVICE = 0 if torch.cuda.is_available() else -1

# Which retrieval strategy answer_with_rag_and_ontology uses for ontology
# triples. Was previously hardcoded inline as `MODE = "hybrid"` inside the
# function body -- pulled out to module config so it's not magic, and can be
# overridden without editing code (e.g. RETRIEVAL_MODE=dense for local testing).
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "hybrid")  # dense | hybrid | cross

TBOX_PATH = str(DATA_DIR / "herb-local-names.json")
LOCAL_RAG_DB_JSONL = str(DATA_DIR / "new_rag_data.jsonl")

# ----------------------------
# NOTE: Thai/Japanese -> English translation (NLLB-200) removed.
# ----------------------------
# The original research code translated the user's query to English before
# ontology/RAG retrieval. In this deployed app, `query_en = query` (no
# translation actually happens) -- see the Known Limitations section of the
# README. Loading NLLB-200 (facebook/nllb-200-distilled-600M) just to never
# call it was adding several hundred MB and real startup time for nothing.
# If real translation is reinstated later, add the model load and the
# translate()/thai_to_english()/english_to_thai() functions back here, and
# actually call thai_to_english() where query_en is assigned below.

# ----------------------------
# LOAD LLAMA MODEL and reranker
# ----------------------------
print("Loading Llama model...")
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def rerank_chunks(query, chunks, top_k=3):
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked[:top_k]]

def rerank_with_ontology(query, chunks, triples, alpha=0.8):
    pairs = [(query, c) for c in chunks]
    q_scores = reranker.predict(pairs)

    triple_text = " ".join(
        f"{t['subject']} {t['predicate']} {t['object']}"
        for t in triples
    )
    t_pairs = [(triple_text, c) for c in chunks]
    t_scores = reranker.predict(t_pairs)

    final = [(c, alpha * q + (1 - alpha) * t) for c, q, t in zip(chunks, q_scores, t_scores)]
    return [c for c, _ in sorted(final, key=lambda x: x[1], reverse=True)[:3]]


tokenizer_llama = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model_llama = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)
pipe_llama = pipeline(
    "text-generation",
    model=model_llama,
    tokenizer=tokenizer_llama,
    torch_dtype=torch.bfloat16,
    device=DEVICE,
    # do_sample=False (explicit): the Llama-3.2-1B-Instruct checkpoint's own
    # generation_config.json ships with do_sample=True by default. Without
    # this line, that default silently wins -- the same query can produce a
    # different answer on every run, with no error and no obvious cause. For
    # a repo meant to let someone else reproduce this project, deterministic
    # output matters: same input -> same output, every time, on any machine.
    # Set do_sample=True instead if you want varied phrasing on repeat runs,
    # but treat that as a deliberate choice, not a default to inherit blind.
    do_sample=False,
    max_new_tokens=120,  # Tried raising this to 220 to test whether longer answers
                         # were being cut off mid-thought. They were, but that wasn't
                         # the real problem: past ~120 tokens the model doesn't finish
                         # a coherent answer, it wanders into unrelated, invented
                         # content (see README Known Limitations). Reverted -- 120
                         # fails by stopping early, which is the less-bad failure mode
                         # than 220's fails-by-hallucinating-further.
    repetition_penalty=1.05,
    no_repeat_ngram_size=3,
)

# ----------------------------
# LOAD TBOX ONTOLOGY TRIPLES
# ----------------------------
def load_ontology_triples(json_path: str):
    print(f"Loading ontology from {json_path}...")
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    clean_triples = []
    print("Parsing JSON structure...")
    for item in data:
        entry = {}
        if "subject" in item:
            entry = {
                "subject": item.get("subject"),
                "predicate": item.get("predicate"),
                "object": item.get("object"),
                "type": "standard",
            }
        if entry.get("subject") and entry.get("predicate") and entry.get("object"):
            clean_triples.append(entry)

    print(f"Raw load complete. Found {len(clean_triples)} triples.")
    return clean_triples

ONTOLOGY_TRIPLES = load_ontology_triples(TBOX_PATH)

def local_name(uri: str) -> str:
    return uri.split("#")[-1].split("/")[-1]

def verbalize_triple(triple):
    """Converts a triple dict into a natural language sentence."""
    s, p, o = triple["subject"], triple["predicate"], triple["object"]
    t_type = triple.get("type", "standard")

    if t_type == "restriction":
        return f"{s} on property {p} takes values from {o}"
    if "label" in p.lower():
        return f"{s} is called {o}"
    return f"{s} {p} {o}"

def expand_triples_for_prompt(triples):
    expanded = []
    for t in triples:
        text = verbalize_triple(t)
        expanded.append(text)
        expanded.append(f"{t['subject']} involves {t['object']}")
    return list(dict.fromkeys(expanded))

# ----------------------------
# EMBEDDING MODEL (BGE-M3)
# ----------------------------
# NOTE: previously loaded TWICE under two names (bge_tokenizer/bge_model for
# ontology triples, rag_tokenizer/rag_model for RAG chunks) -- same model,
# same weights, loaded into memory independently. Now loaded once and shared
# by both embed_text() (below) and the ontology/RAG embedding steps that
# used to have their own near-duplicate get_embedding() function.
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(EMBED_MODEL)

def embed_text(text: str) -> np.ndarray:
    inputs = embed_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        outputs = embed_model(**inputs)
    # mean pooling (correct for bge-m3)
    emb = outputs.last_hidden_state.mean(dim=1)
    emb = F.normalize(emb, p=2, dim=1)
    return emb[0].cpu().numpy()

def embed_query(text):
    return embed_text(text)

def embed_passage(text):
    return embed_text("Represent this passage for retrieval: " + text)

# ----------------------------
# CACHE TBOX EMBEDDINGS
# ----------------------------
TRIPLE_EMB_PATH = str(DATA_DIR / "triple_embeddings_json_herb.npy")
triple_texts = [verbalize_triple(t) for t in ONTOLOGY_TRIPLES]

if os.path.exists(TRIPLE_EMB_PATH):
    print("Loading cached ontology embeddings...")
    triple_embeddings = np.load(TRIPLE_EMB_PATH)
else:
    print("Computing ontology embeddings...")
    triple_embeddings = np.vstack([embed_text(txt) for txt in triple_texts])
    np.save(TRIPLE_EMB_PATH, triple_embeddings)

def retrieve_tbox_dense(query, top_k=5, q_emb=None):
    if q_emb is None:
        q_emb = embed_query(query)
    sims = np.dot(triple_embeddings, q_emb)
    idx = np.argsort(sims)[::-1][:top_k]
    return [ONTOLOGY_TRIPLES[i] for i in idx]

def retrieve_tbox_hybrid(query_en, top_k=5, q_emb=None):
    # 1. Lexical filter (keyword match)
    q_tokens = set(query_en.lower().split())
    candidate_indices = []
    for i, t in enumerate(ONTOLOGY_TRIPLES):
        t_tokens = set(verbalize_triple(t).lower().split())
        if len(q_tokens & t_tokens) >= 1:
            candidate_indices.append(i)

    # 2. Fallback: no keyword match -> consider all triples
    if not candidate_indices:
        candidate_indices = list(range(len(ONTOLOGY_TRIPLES)))

    # 3. Dense scoring using cache. Accepts a precomputed q_emb so callers
    # that already embedded this exact query text (e.g. smart_answer) don't
    # pay for a second forward pass on the same string.
    candidate_embs = triple_embeddings[candidate_indices]
    if q_emb is None:
        q_emb = embed_query(query_en)
    sims = np.dot(candidate_embs, q_emb)

    top_k_local = np.argsort(sims)[::-1][:top_k]
    final_indices = [candidate_indices[i] for i in top_k_local]
    return [ONTOLOGY_TRIPLES[i] for i in final_indices]

def retrieve_tbox_crossencoder(query, top_k=5):
    texts = [verbalize_triple(t) for t in ONTOLOGY_TRIPLES]
    pairs = [(query, t) for t in texts]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(ONTOLOGY_TRIPLES, scores), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked[:top_k]]

# ----------------------------
# LOAD LOCAL RAG VECTOR DB
# ----------------------------
with open(LOCAL_RAG_DB_JSONL, "r", encoding="utf-8") as f:
    local_data = [json.loads(line) for line in f]

texts = [json.loads(d["json"])["text"] for d in local_data]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=80,
    separators=["\n\n", "\n", " ", ""],
)
split_docs = splitter.create_documents(texts)
split_texts = [d.page_content for d in split_docs]

# ----------------------------
# CACHE RAG EMBEDDINGS
# ----------------------------
RAG_EMB_PATH = str(DATA_DIR / "rag_embeddingsnew_herb.npy")

if os.path.exists(RAG_EMB_PATH):
    print("Loading cached RAG embeddings...")
    rag_embeddings = np.load(RAG_EMB_PATH)
else:
    print("Computing RAG embeddings...")
    rag_embeddings = np.array([embed_passage(text) for text in split_texts])
    np.save(RAG_EMB_PATH, rag_embeddings)

def retrieve_locally(query, top_k=5, q_emb=None):
    if q_emb is None:
        q_emb = embed_query(query)
    q_emb = q_emb.reshape(1, -1)
    sims = cosine_similarity(rag_embeddings, q_emb).squeeze()
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [split_texts[i] for i in top_indices]

def ontology_relevance_score(query, q_emb=None):
    if q_emb is None:
        q_emb = embed_query(query)
    sims = np.dot(triple_embeddings, q_emb)
    return float(np.max(sims))

def rag_relevance_score(query, q_emb=None):
    if q_emb is None:
        q_emb = embed_query(query)
    sims = cosine_similarity(rag_embeddings, q_emb.reshape(1, -1)).squeeze()
    return float(np.max(sims))

# ----------------------------
# UTILS
# ----------------------------
def truncate_text(text, max_chars=300):
    return text[:max_chars].strip() + ("..." if len(text) > max_chars else "")

DOMAIN_KEYWORDS = set(
    word.lower()
    for t in ONTOLOGY_TRIPLES
    for word in verbalize_triple(t).split()
    if len(word) > 2
)

CHITCHAT_PHRASES = {
    "hello", "hi", "hey",
    "สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ",
    "หวัดดี", "ดีครับ", "ดีค่ะ",
    "thank you", "thanks", "ขอบคุณ",
    "bye", "goodbye", "ลาก่อน",
}

def is_chitchat_query(query: str) -> bool:
    """
    Was previously: exact-match against CHITCHAT_PHRASES, OR any query
    <= 4 characters. That second rule was too aggressive -- several short
    but legitimate Thai questions are <= 4 characters and would have been
    silently routed away from retrieval. Narrowed to <= 2 characters, which
    catches things like "ok"/single acknowledgements without swallowing
    short real questions. Still a blunt heuristic -- worth swapping for a
    small intent classifier if this goes beyond demo scope.
    """
    q = query.lower().strip()
    if q in CHITCHAT_PHRASES:
        return True
    if len(q) <= 2:
        return True
    return False

# ----------------------------
# MAIN FUNCTION: Query + TBox -> RAG
# ----------------------------
def answer_with_llama_only(query):
    prompt = f"""
คุณเป็นผู้ช่วย AI ที่สุภาพและเป็นมิตร
ตอบคำถามทั่วไปตามธรรมชาติ
คำถาม:
{query}

คำตอบ:
""".strip()

    output = pipe_llama(prompt)[0]["generated_text"]
    return output.split("คำตอบ:")[-1].strip().replace("\n", " ")

def answer_with_rag_and_ontology(query: str, topk_triples=5, topk_rag=5, q_emb=None):
    original_query = query

    # Translation step removed -- see note near the top of this file.
    query_en = query

    if q_emb is None:
        q_emb = embed_query(query_en)

    if RETRIEVAL_MODE == "dense":
        top_triples = retrieve_tbox_dense(query_en, topk_triples, q_emb=q_emb)
    elif RETRIEVAL_MODE == "cross":
        top_triples = retrieve_tbox_crossencoder(query_en, topk_triples)
    else:
        top_triples = retrieve_tbox_hybrid(query_en, topk_triples, q_emb=q_emb)

    expanded_triples = expand_triples_for_prompt(top_triples)
    triple_text_block_trunc = truncate_text(" ".join(expanded_triples), 500)

    # This is a DIFFERENT string than query_en (it includes the triple
    # context), so it genuinely needs its own embedding -- not reusable
    # from q_emb above.
    expanded_query = "From context in Knowledge Graph: " + triple_text_block_trunc + " User question: " + query

    rag_candidates = retrieve_locally(expanded_query, top_k=topk_rag)
    rag_chunks = rerank_chunks(expanded_query, rag_candidates, top_k=5)
    best_rag_chunk = " ".join(rag_chunks)

    prompt = f"""
คุณเป็นผู้เชี่ยวชาญด้านการปลูกและการผลิตมะม่วง
หากข้อมูลใน Ontology หรือ RAG มีความเกี่ยวข้อง ให้ใช้เป็นหลัก
ห้ามสร้างข้อมูลใหม่ที่ขัดแย้งกับ Ontology
ห้ามเชื่อมโยง ห้ามสรุปเกินจากข้อมูลต้นฉบับ
**ห้ามขึ้นบรรทัดใหม่เด็ดขาด**
**ห้ามใช้รายการ ห้ามใช้ bullet points ห้ามมีหมายเลขลำดับ**
รวมคำตอบทั้งหมดให้อยู่ใน 1 บรรทัดเท่านั้น โดยให้ข้อมูลครบถ้วน
หากข้อมูลไม่ครบ ให้ตอบจาก RAG เท่าที่มี โดยไม่เพิ่มข้อมูลใหม่
do not repeat the question in the answer

Ontology triples :
{triple_text_block_trunc}

RAG context:
{best_rag_chunk}

คำถาม:
{original_query}

คำตอบ:
""".strip()

    output = pipe_llama(prompt)[0]["generated_text"]
    answer = output.split("คำตอบ:")[-1].strip().replace("\n", " ")
    if answer.strip() == "" or answer.lower().startswith("none"):
        answer = "{NONE}"

    return {
        "ontology_triples": top_triples,
        "rag_chunks_list": rag_chunks,
        "answer": answer,
    }

def smart_answer(query, topk_triples=5, topk_rag=5):
    # 1. Hard rule: chit-chat -> llama only, no retrieval
    if is_chitchat_query(query):
        return {
            "mode": "llama_only",
            "answer": answer_with_llama_only(query),
            "ontology_triples": [],
        }

    # 2. Semantic relevance check (only for real questions).
    # Embed the query ONCE here and pass it through to both relevance
    # checks and retrieval -- previously this same string was embedded
    # up to 3 separate times (ontology_relevance_score, rag_relevance_score,
    # retrieve_tbox_hybrid) with no caching.
    q_emb = embed_query(query)
    onto_score = ontology_relevance_score(query, q_emb=q_emb)
    rag_score = rag_relevance_score(query, q_emb=q_emb)

    print(f"[DEBUG] Ontology score: {onto_score:.3f} | RAG score: {rag_score:.3f}")

    ONTO_TH = 0.7   # leave as-is, it's working
    RAG_TH = 0.68   # was 0.74 -- too strict for a 14-passage demo corpus

    if onto_score >= ONTO_TH and rag_score >= RAG_TH:
        result = answer_with_rag_and_ontology(query, topk_triples, topk_rag, q_emb=q_emb)
        result["mode"] = "kgrag"
        result["triple_texts"] = " | ".join(
            verbalize_triple(t) for t in result["ontology_triples"]
        )
        result["rag_chunks"] = " | ".join(result["rag_chunks_list"])
        return result

    return {
        "mode": "llama_only",
        "answer": answer_with_llama_only(query),
        "ontology_triples": [],
        "triple_texts": "",
        "rag_chunks": "",
    }

# ----------------------------
# SIMPLE WHILE TRUE LOOP
# ----------------------------
if __name__ == "__main__":
    print("\n--- Mango AI Ready (Type 'exit' to quit) ---")
    while True:
        user_q = input("\nถามคำถาม: ").strip()
        if user_q.lower() in ["exit", "quit"]:
            break

        result = smart_answer(user_q)
        print(f"[MODE: {result['mode']}] {result['answer']}\n")
        print(f"Triples used:{result['triple_texts']}\n")
        print(f"RAG chunks used:{result['rag_chunks']}")