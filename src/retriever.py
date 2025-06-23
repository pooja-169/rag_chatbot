# src/retriever.py

from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import os

# Load the embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get the current directory to load index and chunks correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.idx")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.pkl")

def embed_text(text):
    """
    Converts a string or list of strings into embedding(s).
    """
    return model.encode(text, convert_to_tensor=False)

def retrieve(query, k=3):
    """
    Retrieves the top-k most relevant chunks from the FAISS index based on the query.
    """
    # Load FAISS index
    index = faiss.read_index(INDEX_PATH)

    # Load chunks (original texts corresponding to the index vectors)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    # Embed the query
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)

    # Return the top-k matched chunks
    return [chunks[i] for i in I[0]]

# Optional CLI test
if __name__ == "__main__":
    query = input("Enter a query: ")
    results = retrieve(query)
    print("\nTop results:")
    for i, chunk in enumerate(results, 1):
        print(f"{i}. {chunk.strip()}\n")

def retrieve_from_custom_index(query, chunks, k=5):
    """
    Builds a temporary in-memory index from provided chunks,
    and retrieves top-k relevant chunks for the query.
    """
    # Embed all chunks
    embeddings = model.encode(chunks, convert_to_tensor=False)
    dim = embeddings[0].shape[0]

    # Create temporary FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # Embed the query
    query_embedding = model.encode([query])

    # Search and return top-k chunks
    D, I = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in I[0]]
