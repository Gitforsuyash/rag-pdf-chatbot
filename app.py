import streamlit as st
# Ensure that utils/pdf.py exists and contains extract_text_from_pdf
from utils.pdf_loader import extract_text_from_pdf
from utils.vector_store import create_vector_store

st.set_page_config(page_title="📄 RAG Chatbot for PDFs")
st.title("📄 RAG Chatbot for PDFs")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success("✅ PDF uploaded successfully.")
    
    # Extract text from PDF
    text = extract_text_from_pdf(uploaded_file)
    st.write(f"Extracted {len(text)} characters.")

    # Create FAISS vector store
    vector_db = create_vector_store(text)
    st.success("✅ Vector store created using FAISS.")

    # Chat interface
    st.header("💬 Ask questions about your PDF")
    user_question = st.text_input("Enter your question")

    if user_question:
        docs = vector_db.similarity_search(user_question, k=3)
        st.subheader("🔍 Top Matches")
        for i, doc in enumerate(docs):
            st.markdown(f"**Match {i+1}:**")
            st.write(doc.page_content[:500000])  # Preview first 500000 characters
