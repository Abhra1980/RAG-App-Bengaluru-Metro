import streamlit as st
from dotenv import load_dotenv
from utils.dataload import read_text_file
from utils.datastore import save_text_to_csv
from utils.datastore import save_text_to_json
import os
from utils.clening import clean_text
from utils.chunking import chunk_text
from utils.embedding import generate_embeddings
from utils.embedding import create_embeddings
from utils.faisssave import save_faiss_index
from utils.faissload import load_faiss_index_from_dir
from utils.search_threshold import results_to_paragraph
from utils.search_threshold import retrieve_context
from utils.search_threshold import generate_completion
from utils.search_threshold import generate_embeddings
from utils.search_threshold import search_faiss
from utils.prompt_completion import build_prompt
from pathlib import Path
DATA_DIR = (Path(__file__).resolve().parent / "faiss_store").resolve()
dir_path = DATA_DIR
dir_path = dir_path
index, meta = load_faiss_index_from_dir(dir_path, "metro_index.faiss", "metro_meta.jsonl")

#query = "Bengaluru's Namma Metro project"

st.title("RAG App – Bengaluru's Namma Metro project")
st.write("Ask questions grounded in Bengaluru's Namma Metro project")
query = st.text_input("Enter your question here")

if query:
    DATA_DIR = (Path(__file__).resolve().parent / "faiss_store").resolve()
    dir_path = DATA_DIR
    dir_path = dir_path
    index, meta = load_faiss_index_from_dir(dir_path, "metro_index.faiss", "metro_meta.jsonl")
    data_output = search_faiss(index, query, top_k=6)
    THRESHOLD = 0.30  # 50%
    paragraph = results_to_paragraph(data_output, meta, threshold=THRESHOLD)
    context_chunks = retrieve_context(data_output, meta, top_n=3, threshold=THRESHOLD)
    prompt = build_prompt(query, context_chunks)
    answer = generate_completion(prompt)
    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved Chunks"):
        for chunk in context_chunks:
            st.markdown(f"- {chunk}")