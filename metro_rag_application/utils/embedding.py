# 4) Embeddings via Euron API
#now generate embeddings for each chunk using euron api
import requests
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
load_dotenv() 
EURI_API_KEY = os.getenv("EURI_API_KEY")

def generate_embeddings(text):
    url = "https://api.euron.one/api/v1/euri/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"{EURI_API_KEY}"
    }
    payload = {
        "input": text,
        "model": "text-embedding-3-small"
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    
    embedding = np.array(data['data'][0]['embedding'])
    
    return embedding


def create_embeddings(all_chunks: List[str]):
    emb_list: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(all_chunks):
        vec = generate_embeddings(chunk)
        emb_list.append(vec.astype("float32"))
        meta.append({"id": idx, "text": chunk})
    return emb_list, meta
