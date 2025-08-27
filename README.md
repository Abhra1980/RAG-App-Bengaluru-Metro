# RAG-App-Bengaluru-Metro
This project builds a tiny Retrieval-Augmented Generation (RAG) pipeline around a Namma Metro knowledge base.
Overview
--------
This project is a minimal Retrieval-Augmented Generation (RAG) demo built around
a Bangalore Namma Metro knowledge base. It demonstrates a complete pipeline:
1) clean + chunk source text, 2) generate embeddings with the Euron API,
3) build a FAISS vector index (cosine similarity via inner-product on normalized
vectors), 4) retrieve top matches for a query with a score threshold, and
5) optionally call a chat completion endpoint **only** when retrieval confidence
is high enough.

If results don’t meet the configured threshold, the app returns the message:
“I am not trend on given data” (you may want to change this to “I am not trained
on the given data”).


Features
--------
- Text cleaning with NLTK (lowercasing, punctuation removal, stopwords, lemmatization)
- Character-based chunking with overlap (defaults: 500 max chars, 100 overlap)
- Embeddings via Euron API (example: `text-embedding-3-small`)
- FAISS IndexFlatIP for cosine-like similarity (uses L2-normalized vectors)
- Threshold-gated retrieval (default 0.50); prevents weak/irrelevant context
- Conditional LLM call to Euron chat completions only when matches pass threshold
- Saves and reloads index/metadata: `.faiss` and `.jsonl`
- Fully local index and deterministic retrieval; replace the dataset with your own


Requirements
-----------
- Python 3.9+
- pip packages:
  - pandas, numpy, requests
  - nltk
  - faiss-cpu  (or faiss-gpu if you know what you’re doing)

Install example:
    pip install pandas numpy requests nltk faiss-cpu

First run, NLTK will download resources (punkt, stopwords, wordnet, omw-1.4).


Environment Variables
---------------------
**Do NOT hardcode your API key in code.** Set it via environment variable:

Linux / macOS:
    export EURI_API_KEY="YOUR_KEY"

Windows (PowerShell):
    $Env:EURI_API_KEY="YOUR_KEY"

The script reads it with:
    EURI_API_KEY = os.getenv("EURI_API_KEY", "")


Quick Start
-----------
1) Clone or copy the script into your workspace.
2) Create and activate a virtual environment (recommended).
3) Install requirements.
4) Set `EURI_API_KEY` in your shell.
5) Run the script:
    python namma_metro_rag_demo.py

On first run it will:
- Clean and chunk the inlined Namma Metro dataset
- Create embeddings for each chunk via the Euron API
- Build and save FAISS index: `index_vecass1.faiss`
- Save metadata: `meta_vecass1.jsonl`
- Run a sample query:
    "inaugurated Hebbagodi Metro station on the Yellow Line"

If retrieved scores are >= 0.50, it will construct a prompt from the top
context chunks and call `generate_completion`. Otherwise you’ll see the
fallback message.


How It Works (Pipeline)
-----------------------
1) **Ingest**: a dataset string is saved to `namma_metro_dataset.csv` (1 row).
2) **Clean**: lowercasing, punctuation removal, lemmatization, stopword filtering.
3) **Chunk**: fixed-size character windows (500) with 100-character overlap.
4) **Embed**: POST to Euron embeddings endpoint with model `text-embedding-3-small`.
5) **Index**: stack vectors, L2-normalize, add to FAISS IndexFlatIP; persist to disk.
6) **Search**: encode the query, L2-normalize, `index.search` for top-k results.
7) **Gate**: accept only results with `score >= THRESHOLD` (default 0.50).
8) **RAG**: build a prompt from top-N matches (default 3) and call chat completions.
9) **Answer**: print the prompt and returned model answer.


Key Files Written
-----------------
- namma_metro_dataset.csv   (the cleaned dataset in CSV form)
- index_vecass1.faiss       (FAISS index on embeddings)
- meta_vecass1.jsonl        (one JSON object per chunk: id + text)


Configuration Knobs
-------------------
- `max_char` and `overlap` in `chunk_text(...)`
- `THRESHOLD` for minimum similarity score (0.0–1.0 typical)
- `top_k` in `search_faiss(...)` and `top_n` in `retrieve_context(...)`
- Model names / endpoints for embeddings and chat
- Replace the inlined dataset string with your own corpus (TXT, CSV, etc.)


Security Notes
--------------
- Remove hardcoded keys from your script; prefer environment variables.
- Avoid logging sensitive data (headers, payloads).
- Consider `.env` files plus `python-dotenv` if you prefer local-only storage.


Troubleshooting & Tips
----------------------
- **404 Route Not Found** on chat completions:
  Double-check the base URL and path you’re using for the chat completions API.
  Some deployments use a different path or omit `/alpha`. Verify model names and
  endpoints in your provider’s docs.
- **NLTK resource errors**:
  Ensure `nltk.download(...)` calls are present (they are in this script) and
  that your environment has internet access the first time you run it.
- **FAISS install issues** on Windows:
  Prefer `faiss-cpu` via pip in a clean virtual environment.
- **Cross-platform file paths**:
  Use `pathlib.Path` instead of raw strings: `Path("data") / "file.jsonl"`.
- **“I am not trend on given data”**:
  That string is intentional to show threshold gating. Change to a friendlier
  message if you like.


Extending This Project
----------------------
- Swap in your own documents (TXT, CSV, PDF excerpted text) before embedding.
- Persist the cleaned/chunked corpus to disk for repeatable runs.
- Add caching of embeddings to avoid re-embedding unchanged text.
- Switch to a retrieval framework (LlamaIndex/LangChain) if you want chains/agents.
- Build a Streamlit UI for uploads, search results, and the final answer.
