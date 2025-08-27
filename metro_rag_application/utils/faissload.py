# Load FAISS index and metadata
import os, json, faiss
from typing import List, Dict, Any, Tuple

def load_faiss_index_from_dir(
    dir_path: str,
    index_filename: str = "index_vecass1.faiss",
    meta_filename: str  = "meta_vecass1.jsonl",
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    index_path = os.path.join(dir_path, index_filename)
    meta_path  = os.path.join(dir_path, meta_filename)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    index = faiss.read_index(index_path)

    meta_list: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta_list.append(json.loads(line))

    if index.ntotal != len(meta_list):
        print(f"⚠️ Warning: index vectors ({index.ntotal}) != metadata records ({len(meta_list)})")

    print(f"Loaded index: {index_path} | metadata records: {len(meta_list)}")
    return index, meta_list


