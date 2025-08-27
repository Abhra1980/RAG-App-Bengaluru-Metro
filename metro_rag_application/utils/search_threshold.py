# 6) Search + threshold logic
# -----------------------------
from utils.embedding import generate_embeddings
import faiss
import numpy as np
from typing import List, Dict, Tuple, Any
import os
import requests
 # loads variables from .env into environment
#load_dotenv() 
EURI_API_KEY = os.getenv("EURI_API_KEY")

THRESHOLD = 0.30  # 50%

def search_faiss(index: faiss.IndexFlatIP, query: str, top_k: int = 6):
    q = generate_embeddings(query).reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, top_k)
    return scores, idxs


def results_to_paragraph(
    data_output: Tuple[np.ndarray, np.ndarray],
    meta: List[Dict[str, Any]],
    threshold: float = THRESHOLD
):
    scores, idxs = data_output
    scores = np.asarray(scores).ravel().astype(float)
    idxs = np.asarray(idxs).ravel().astype(int)

    meta_by_id = {int(m["id"]): m["text"] for m in meta}

    selected: List[str] = []
    for i, s in zip(idxs, scores):
        if s >= threshold and i in meta_by_id:
            selected.append(meta_by_id[i])

    # Custom message per your requirement:
    if not selected:
        return "I am not trend on given data"

    return " ".join(selected)

def retrieve_context(
    data_output: Tuple[np.ndarray, np.ndarray],
    meta: List[Dict[str, Any]],
    top_n: int = 3,
    threshold: float = THRESHOLD
):
    scores, idxs = data_output
    scores = np.asarray(scores).ravel().astype(float)
    idxs = np.asarray(idxs).ravel().astype(int)

    meta_by_id = {int(m["id"]): m["text"] for m in meta}

    pairs = [(i, s) for i, s in zip(idxs, scores) if s >= threshold and i in meta_by_id]
    pairs = pairs[:top_n]

    return [(meta_by_id[i], float(s)) for i, s in pairs]



# 7) Prompt + completion (only if threshold passed)

def generate_completion(prompt, model="gpt-4.1-nano"):
    url = "https://api.euron.one/api/v1/euri/chat/completions"
    headers = {
        #"Authorization": f"Bearer {EURI_API_KEY}",
        "Authorization": f"{EURI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.3
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']
	
	
