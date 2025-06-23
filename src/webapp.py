# src/webapp.py

import streamlit as st
from scraper import extract_text_from_url
from preprocess import chunk_text
from retriever import retrieve_from_custom_index
from src.gemini_generator import generate_answer_from_chunks


st.set_page_config(page_title="Webpage RAG Chatbot", layout="centered")

st.title("🔍 Webpage-Powered Chatbot with Gemini")
st.markdown("Enter a website URL below. The bot will extract content and let you ask questions about it.")

# Step 1: User enters a URL
url = st.text_input("🌐 Enter a website URL to load content")

if url:
    with st.spinner("🔄 Fetching and extracting content..."):
        text = extract_text_from_url(url)

    if not text or len(text.strip()) < 50:
        st.error("Failed to extract meaningful content from the provided URL.")
    else:
        st.success("✅ Text extracted successfully!")

        # Show a preview
        st.markdown("### 📄 Extracted Text Preview:")
        st.code(text[:1000] + "..." if len(text) > 1000 else text)

        # Step 2: Chunk the text
        chunks = chunk_text(text)

        # Step 3: Ask questions
        st.markdown("### ❓ Ask a Question")
        user_query = st.text_input("Ask something based on the above page:")

        if user_query:
            with st.spinner("🤖 Generating answer using Gemini..."):
                top_chunks = retrieve_from_custom_index(user_query, chunks, k=5)
                answer = generate_answer_from_chunks(user_query, top_chunks)

            st.markdown("### 💬 Answer:")
            st.success(answer)
