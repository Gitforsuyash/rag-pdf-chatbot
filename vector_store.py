from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def create_vector_store(text, save_path="faiss_index"):
    # No API key needed here!
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    documents = [Document(page_content=text)]
    vectordb = FAISS.from_documents(documents, embeddings)
    vectordb.save_local(save_path)
    return vectordb
