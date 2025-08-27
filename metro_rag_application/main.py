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
DATA_DIR = (Path(__file__).resolve().parent / "data").resolve()
dir_path = DATA_DIR


import faiss
 # loads variables from .env into environment
load_dotenv() 
EURI_API_KEY = os.getenv("EURI_API_KEY")
#data loading
#dir_path = r"C:\Data Science\Assignments\Final\metro_rag_application\data"
dir_path = dir_path
dataset = read_text_file(dir_path, "dataset.txt")

#data saving
#dir_path = r"C:\Data Science\Assignments\Final\metro_rag_application\data"
save_text_to_csv(dataset, dir_path, "namma_metro_dataset.csv")

save_text_to_json(dataset, dir_path, "namma_metro_dataset.json")

#clean text
cleaned_data = clean_text(dataset)

#Chunking text
chunks = chunk_text(cleaned_data, max_char=500, overlap=100)

#Embeddings via Euron API
emb_list, meta = create_embeddings(chunks)

#FAISS index save/load (cosine via IP)
DATA_DIR = (Path(__file__).resolve().parent / "faiss_store").resolve()
dir_path = DATA_DIR
save_faiss_index(
    emb_list,
    meta,
    #dir_path=r"C:\\Data Science\\Assignments\\Final\\metro_rag_application\\faiss_store",
    dir_path = dir_path,
    index_filename="metro_index.faiss",
    meta_filename="metro_meta.jsonl"
)

# load (cosine via IP)
#dir_path = r"C:\\Data Science\\Assignments\\Final\\metro_rag_application\\faiss_store"
dir_path = dir_path
index, meta = load_faiss_index_from_dir(dir_path, "metro_index.faiss", "metro_meta.jsonl")

# 8) Demo run
# -----------------------------
query = "Bengaluru's Namma Metro project"

data_output = search_faiss(index, query, top_k=6)

# First, check the thresholded paragraph:
THRESHOLD = 0.30  # 50%
paragraph = results_to_paragraph(data_output, meta, threshold=THRESHOLD)
print("\n--- Thresholded Paragraph or Message ---\n", paragraph, "\n")
# We have relevant context; proceed to build a prompt and call the model.
context_chunks = retrieve_context(data_output, meta, top_n=3, threshold=THRESHOLD)
# Safety: if, for some reason, context_chunks is empty, also show the same message.
prompt = build_prompt(query, context_chunks)
print("\n--- Prompt Sent to Model ---\n", prompt, "\n")
answer = generate_completion(prompt)
print("--- Model Answer ---\n", answer)