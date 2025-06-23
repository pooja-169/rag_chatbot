from retriever import retrieve
import json
import requests
import os
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def generate_answer_from_chunks(query, chunks=None):
    context_chunks = chunks if chunks else retrieve(query)
    context = "\n\n".join(context_chunks)

    prompt = (
        f"Use the following context to answer the question clearly and concisely.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response_data = response.json()

    try:
        return response_data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"[Error] {str(e)}\nRaw response:\n{json.dumps(response_data, indent=2)}"
