import torch
from rdflib import Graph, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM, AutoModel
import numpy as np
import torch.nn.functional as F
import json
from sklearn.metrics.pairwise import cosine_similarity
import time
import pandas as pd
import os
from pythainlp.tokenize import word_tokenize
from nltk.util import ngrams
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer
from rouge import Rouge
from rouge_score import rouge_scorer
from sentence_transformers import CrossEncoder
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
import re

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

TBOX_PATH = str(DATA_DIR / "herb-local-names.json")
LOCAL_RAG_DB_JSONL = str(DATA_DIR / "new_rag_data.jsonl")

# ----------------------------
# LOAD MULTILINGUAL TRANSLATION MODEL (NLLB-200)
# ----------------------------
print("Loading multilingual translation model (NLLB-200)...")
nllb_model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)
translator = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device=DEVICE,
)

def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    outputs = translator(
        text,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_length=400,
    )
    return outputs[0]["generated_text"]


def thai_to_english(text):
    return translate(text, src_lang="tha_Latn", tgt_lang="eng_Latn")

def english_to_thai(text):
    return translate(text, src_lang="eng_Latn", tgt_lang="tha_Latn")

def japanese_to_english(text):
    """
    Translates text if it contains Japanese characters using the loaded NLLB model.
    """
    text = str(text)
    # Simple check for Japanese unicode ranges (Hiragana, Katakana, CJK Unified Ideographs)
    is_japanese = any("\u3040" <= char <= "\u309f" or 
                      "\u30a0" <= char <= "\u30ff" or 
                      "\u4e00" <= char <= "\u9faf" for char in text)
    
    if is_japanese:
        # jpn_Jpan is the NLLB code for Japanese
        return translate(text, src_lang="jpn_Jpan", tgt_lang="eng_Latn")
    return text

# ----------------------------
# LOAD LLAMA MODEL and reranker
# ----------------------------
print("Loading Llama model...")
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    # "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def rerank_chunks(query, chunks, top_k=3):
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )
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

    final = [
        (c, alpha*q + (1-alpha)*t)
        for c, q, t in zip(chunks, q_scores, t_scores)
    ]

    return [c for c, _ in sorted(final, key=lambda x: x[1], reverse=True)[:3]]


tokenizer_llama = AutoTokenizer.from_pretrained(MODEL_NAME, token = HF_TOKEN)
model_llama = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)
pipe_llama = pipeline(
    "text-generation",
    model=model_llama,
    tokenizer=tokenizer_llama,
    torch_dtype=torch.bfloat16,
    device=DEVICE,
    temperature=0.3,
    max_new_tokens=120,
    top_k=5,
    repetition_penalty=1.05,
    no_repeat_ngram_size=3,    # blocks exact 3-word-phrase loops directly
)

# ----------------------------
# LOAD TBOX ONTOLOGY TRIPLES
# ----------------------------
def load_ontology_triples(json_path: str):
    print(f"Loading ontology from {json_path}...")
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    clean_triples = []
    
    # 1. First Pass: Load Raw Data & Collect text to translate
    print("Parsing JSON structure...")
    for item in data:
        entry = {}
        # Check type and extract raw values
        if "subject" in item:
            entry = {
                "subject": item.get("subject"),
                "predicate": item.get("predicate"),
                "object": item.get("object"),
                "type": "standard"
            }
        # elif "class" in item:
        #     entry = {
        #         "subject": item.get("class"),
        #         "predicate": item.get("on property"),
        #         "object": item.get("all values from"),
        #         "type": "restriction"
        #     }
            
        if entry.get("subject") and entry.get("predicate") and entry.get("object"):
            clean_triples.append(entry)

    print(f"Raw load complete. Found {len(clean_triples)} triples.")
    return clean_triples

ONTOLOGY_TRIPLES = load_ontology_triples(TBOX_PATH)
def local_name(uri: str) -> str:
    return uri.split("#")[-1].split("/")[-1]

def verbalize_triple(triple):
    """
    Converts a triple dictionary into a natural language sentence 
    based on its type.
    """
    s = triple["subject"]
    p = triple["predicate"]
    o = triple["object"]
    t_type = triple.get("type", "standard")

    if t_type == "restriction":
        # Format: "Basic Work on property Purpose takes values from Farming"
        return f"{s} on property {p} takes values from {o}"
    else:
        # Format: "Subject Predicate Object"
        # Optional: Make 'label' sound better
        if "label" in p.lower():
             return f"{s} is called {o}"
        return f"{s} {p} {o}"

def expand_triples_for_prompt(triples):
    expanded = []
    for t in triples:
        # Use the smart verbalization logic defined above
        text = verbalize_triple(t)
        expanded.append(text)
        
        # Add extra semantic expansions if needed
        s = t["subject"]
        o = t["object"]
        expanded.append(f"{s} involves {o}")

    return list(dict.fromkeys(expanded))

# ----------------------------
# EMBEDDING MODEL (BGE-M3)
# ----------------------------
bge_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
bge_model = AutoModel.from_pretrained(EMBED_MODEL)

# def embed_text(text: str) -> np.ndarray:
#     inputs = bge_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = bge_model(**inputs)
#     embeddings = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
#     return embeddings[0].cpu().numpy()
def embed_text(text: str) -> np.ndarray:
    inputs = bge_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = bge_model(**inputs)

    # mean pooling (CORRECT for bge-m3)
    emb = outputs.last_hidden_state.mean(dim=1)
    emb = F.normalize(emb, p=2, dim=1)
    return emb[0].cpu().numpy()

def embed_query(text):
    return embed_text(text)

def embed_passage(text):
    return embed_text("Represent this passage for retrieval: " + text)



# Embed TBox triples
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

def get_relevant_triples(query: str, top_k=5):
    # q_emb = embed_text(query)
    q_emb = embed_query(query)
    sims = np.dot(triple_embeddings, q_emb)
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [ONTOLOGY_TRIPLES[i] for i in top_idx]




def retrieve_tbox_dense(query, top_k=5):
    q_emb = embed_text(query)
    sims = np.dot(triple_embeddings, q_emb)
    idx = np.argsort(sims)[::-1][:top_k]
    return [ONTOLOGY_TRIPLES[i] for i in idx]

def lexical_filter(query, triples, min_overlap=1):
    q_tokens = set(query.lower().split())
    candidates = []

    for t in triples:
        text = verbalize_triple(t).lower()
        t_tokens = set(text.split())
        if len(q_tokens & t_tokens) >= min_overlap:
            candidates.append(t)

    return candidates

# def retrieve_tbox_hybrid(query_en, top_k=5, pre_k=30):
#     # Step 1: lexical filter
#     filtered = lexical_filter(query_en.lower(), ONTOLOGY_TRIPLES)

#     if len(filtered) == 0:
#         filtered = ONTOLOGY_TRIPLES  # fallback

#     # Step 2: dense embedding
#     texts = [verbalize_triple(t) for t in filtered]
#     embs = np.vstack([embed_text(t) for t in texts])

#     q_emb = embed_text(query_en)
#     sims = np.dot(embs, q_emb)

#     idx = np.argsort(sims)[::-1][:top_k]
#     return [filtered[i] for i in idx]

def retrieve_tbox_hybrid(query_en, top_k=5):
    # 1. Lexical Filter (Keyword Match)
    # Get indices of triples that match keywords
    q_tokens = set(query_en.lower().split())
    candidate_indices = []
    
    for i, t in enumerate(ONTOLOGY_TRIPLES):
        text = verbalize_triple(t).lower()
        t_tokens = set(text.split())
        if len(q_tokens & t_tokens) >= 1: # Min overlap
            candidate_indices.append(i)

    # 2. Fallback: If no keywords match, consider ALL triples
    if not candidate_indices:
        candidate_indices = list(range(len(ONTOLOGY_TRIPLES)))

    # 3. Dense Scoring using CACHE (Fast!)
    # We only look at the rows in triple_embeddings that match our candidates
    candidate_embs = triple_embeddings[candidate_indices] 
    
    q_emb = embed_query(query_en)
    
    # Calculate Similarity only for candidates
    sims = np.dot(candidate_embs, q_emb)
    
    # Get top K relative to the candidate list
    top_k_local_indices = np.argsort(sims)[::-1][:top_k]
    
    # Map back to original global indices
    final_indices = [candidate_indices[i] for i in top_k_local_indices]
    
    return [ONTOLOGY_TRIPLES[i] for i in final_indices]
def retrieve_tbox_crossencoder(query, top_k=5):
    texts = [verbalize_triple(t) for t in ONTOLOGY_TRIPLES]
    pairs = [(query, t) for t in texts]

    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(ONTOLOGY_TRIPLES, scores),
        key=lambda x: x[1],
        reverse=True
    )

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
    # separators=["\n\n", "\n", ".", "!", "?", " "]
    # separators=[".", "!", "?", "\n"]
    separators=["\n\n", "\n", " ", ""]
)

split_docs = splitter.create_documents(texts)
split_texts = [d.page_content for d in split_docs]

# Embed RAG texts
rag_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
rag_model = AutoModel.from_pretrained(EMBED_MODEL)

def get_embedding(text) -> np.ndarray:
    # inputs = rag_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # with torch.no_grad():
    #     outputs = rag_model(**inputs)
    # return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    inputs = rag_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = rag_model(**inputs)

    # mean pooling (CORRECT for bge-m3)
    emb = outputs.last_hidden_state.mean(dim=1)
    emb = F.normalize(emb, p=2, dim=1)
    return emb[0].cpu().numpy()



# ----------------------------
# CACHE RAG EMBEDDINGS
# ----------------------------
RAG_EMB_PATH = str(DATA_DIR / "rag_embeddingsnew_herb.npy")


if os.path.exists(RAG_EMB_PATH):
    print("Loading cached RAG embeddings...")
    rag_embeddings = np.load(RAG_EMB_PATH)
else:
    print("Computing RAG embeddings...")
    # rag_embeddings = np.array([get_embedding(text) for text in split_texts])
    rag_embeddings = np.array([embed_passage(text) for text in split_texts])
    np.save(RAG_EMB_PATH, rag_embeddings)

def retrieve_locally(query, top_k=5):
    # q_emb = get_embedding(query).reshape(1, -1)
    q_emb = embed_query(query).reshape(1, -1)
    sims = cosine_similarity(rag_embeddings, q_emb).squeeze()
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [split_texts[i] for i in top_indices]

def ontology_relevance_score(query):
    q_emb = embed_query(query)
    sims = np.dot(triple_embeddings, q_emb)
    return float(np.max(sims))

def rag_relevance_score(query):
    q_emb = embed_query(query).reshape(1, -1)
    sims = cosine_similarity(rag_embeddings, q_emb).squeeze()
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
def is_chitchat_query(query: str) -> bool:
    q = query.lower().strip()

    # greetings / small talk
    CHITCHAT = {
        "hello", "hi", "hey",
        "สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ",
        "หวัดดี", "ดีครับ", "ดีค่ะ",
        "thank you", "thanks", "ขอบคุณ",
        "bye", "goodbye", "ลาก่อน"
    }

    if q in CHITCHAT:
        return True

    # very short non-informational queries
    #fix this later
    if len(q) <= 4:
        return True

    return False



# ----------------------------
# MAIN FUNCTION: Query + TBox -> RAG
# ----------------------------
# ----------------------------
# CONFIGURATION
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

def answer_with_rag_and_ontology(query: str, topk_triples=5, topk_rag=5):
    original_query = query

    # Step 1: Translate query to English for ontology search
    # query_en = thai_to_english(query)
    query_en = query


    # Step 2: Retrieve top ontology triples (in English)
    # top_triples = get_relevant_triples(query_en, top_k=topk_triples)
    MODE = "hybrid"  # dense | hybrid | cross

    if MODE == "dense":
        top_triples = retrieve_tbox_dense(query_en, topk_triples)
    elif MODE == "hybrid":
        top_triples = retrieve_tbox_hybrid(query_en, topk_triples)
    elif MODE == "cross":
        top_triples = retrieve_tbox_crossencoder(query_en, topk_triples)


    expanded_triples = expand_triples_for_prompt(top_triples)
    triple_text_block = " ".join(expanded_triples)
    triple_text_block_trunc = truncate_text(triple_text_block, 500)

    expanded_query = "From context in Knowledge Graph: " + triple_text_block_trunc + " User question: " + query
    # print(expanded_query)


    # Step 5: Retrieve RAG chunks
    # rag_candidates = retrieve_locally(original_query, top_k=topk_rag)

    # rag_chunks = rerank_chunks(original_query, rag_candidates, top_k=5)
    rag_candidates = retrieve_locally(expanded_query, top_k=topk_rag)

    rag_chunks = rerank_chunks(expanded_query, rag_candidates, top_k=5)

    best_rag_chunk = " ".join(rag_chunks)

    def log_triples(triples):
        return " | ".join([verbalize_triple(t) for t in triples])


    # Step 6: Build prompt for Llama
# maybe used within prompt
# Ontology triples :
# {triple_text_block_trunc}

# RAG context:
# {" ".join(rag_chunks_trunc)}


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
    
    print(prompt)

    output = pipe_llama(prompt)[0]["generated_text"]
    answer = output.split("คำตอบ:")[-1].strip().replace("\n", " ")
    if answer.strip() == "" or answer.lower().startswith("none"):
        answer = "{NONE}"

    return {
        "ontology_triples": top_triples,
        "rag_chunks_list": rag_chunks,
        "answer": answer
    }

def smart_answer(query, topk_triples=5, topk_rag=5):

    # 1️⃣ hard rule: chit-chat → llama
    if is_chitchat_query(query):
        return {
            "mode": "llama_only",
            "answer": answer_with_llama_only(query),
            "ontology_triples": []
        }

    # 2️⃣ semantic relevance (ONLY for real questions)
    onto_score = ontology_relevance_score(query)
    rag_score = rag_relevance_score(query)

    print(f"[DEBUG] Ontology score: {onto_score:.3f} | RAG score: {rag_score:.3f}")

    ONTO_TH = 0.7    # leave as-is, it's working
    RAG_TH = 0.68    # was 0.74 — too strict for a 14-passage demo corpus
    #change value here
    # ONTO_TH = 0
    # RAG_TH = 0

    if onto_score >= ONTO_TH and rag_score >= RAG_TH:

        result = answer_with_rag_and_ontology(query, topk_triples, topk_rag)

        result["mode"] = "kgrag"
        result["triple_texts"] = " | ".join(
            verbalize_triple(t) for t in result["ontology_triples"]
        )

        result["rag_chunks"] = " | ".join(
            result["rag_chunks_list"]
        )

        return result

    else:
        return {
            "mode": "llama_only",
            "answer": answer_with_llama_only(query),
            "ontology_triples": [],
            "triple_texts": "",
            "rag_chunks": ""
        }



# ----------------------------
# SIMPLE WHILE TRUE LOOP
# ----------------------------
if __name__ == "__main__":
    print("\n--- Mango AI Ready (Type 'exit' to quit) ---")
    while True:
        user_q = input("\nถามคำถาม: ").strip()
        if user_q.lower() in ['exit', 'quit']:
            break

        result = smart_answer(user_q)
        print(f"[MODE: {result['mode']}] {result['answer']}\n")
        print(f"Triples used:{result['triple_texts']}\n")
        print(f"RAG chunks used:{result['rag_chunks']}")