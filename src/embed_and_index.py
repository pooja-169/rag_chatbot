# src/embed_and_index.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from preprocess import extract_text_from_pdf, chunk_text

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index():
    text = extract_text_from_pdf("data/example.pdf")
    chunks = chunk_text(text)

    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # Save index and chunks
    faiss.write_index(index, "faiss_index.idx")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Index built and saved!")

if __name__ == "__main__":
    build_index()
