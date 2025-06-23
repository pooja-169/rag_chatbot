# src/generator.py

from transformers import pipeline
from retriever import retrieve

generator = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_answer(query):
    contexts = retrieve(query)
    context = "\n".join(contexts)
    prompt = (
    f"Use the following context extracted from a PDF document to answer the question.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {query}\n\n"
    f"Answer:"
)

    response = generator(prompt, max_length=128, do_sample=False)
    return response[0]['generated_text']

if __name__ == "__main__":
    print(generate_answer("What is the sun?"))
