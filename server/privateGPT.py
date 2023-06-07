import streamlit as st
import os
from typing import List
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import requests
import streamlit as st
from typing import List
from langchain.document import Document

load_dotenv()

# Set Streamlit configuration
st.set_page_config(page_title="LangChain Demo")

CHROMA_API_HOST = os.environ.get("CHROMA_API_HOST")  # Update with your Chroma API host
CHROMA_API_PORT = os.environ.get("CHROMA_API_PORT")  # Update with your Chroma API port

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
model_type = os.environ.get("MODEL_TYPE")
model_path = os.environ.get("MODEL_PATH")
model_n_ctx = os.environ.get("MODEL_N_CTX")
llm = None

def load_documents(files: List[st.uploaded_file_manager.UploadedFile]) -> List[Document]:
    documents = []
    for file in files:
        document = Document(file.name, file.getvalue())
        documents.append(document)
    return documents


def get_answer(query: str):
    global llm

    if llm is None:
        qa = RetrievalQA(
            model_type=model_type,
            model_path=model_path,
            model_n_ctx=int(model_n_ctx),
            question_embedding_model=HuggingFaceEmbeddings(embeddings_model_name)
        )

        llm = qa.llm
        llm.load()

    answer = llm.answer(query)
    source_data = llm.get_data(query)

    return query, answer, source_data

def main():
    st.title("Private GPT")
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)
    documents = load_documents(uploaded_files) if uploaded_files else []
    
    if documents:
        st.markdown("**Uploaded Documents:**")
        for document in documents:
            st.markdown(f"- {document.name}")
    else:
        st.markdown("No documents uploaded.")

if __name__ == "__main__":
    main()
