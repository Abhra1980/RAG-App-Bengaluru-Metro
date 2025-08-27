import os
import faiss
import json
import numpy as np
from typing import List, Dict, Any


def save_faiss_index(
    emb_list: List[np.ndarray],
    meta: List[Dict[str, Any]],
    dir_path: str,
    index_filename: str = "index_vecass1.faiss",
    meta_filename: str = "meta_vecass1.jsonl"
):
    # Ensure directory exists
    os.makedirs(dir_path, exist_ok=True)

    # Build full paths
    index_path = os.path.join(dir_path, index_filename)
    meta_path = os.path.join(dir_path, meta_filename)

    # Build FAISS index
    xb = np.vstack(emb_list).astype("float32")
    faiss.normalize_L2(xb)
    dim = xb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(xb)
    faiss.write_index(index, index_path)

    # Save metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        for item in meta:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ FAISS index saved -> {index_path}")
    print(f"✅ Metadata saved   -> {meta_path} (records: {len(meta)})")
