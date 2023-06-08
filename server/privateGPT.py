import streamlit as st
import os
from typing import List
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import GPT4All

load_dotenv()

# Set Streamlit configuration
st.set_page_config(page_title="LangChain Demo")

# Define the Chroma settings
# from chromadb.config import Settings

# Define the Chroma settings
# CHROMA_SETTINGS = Settings(
#     chroma_db_impl='duckdb+parquet',
#     persist_directory="db/",
#     anonymized_telemetry=False,
#     chroma_api_impl='rest'
# )

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get("PERSIST_DIRECTORY")
model_type = os.environ.get("MODEL_TYPE")
model_path = os.environ.get("MODEL_PATH")
model_n_ctx = os.environ.get("MODEL_N_CTX")
llm = None

class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if "text/html content not found in email" in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        documents = loader.load()
        return documents[0] if documents else None
    else:
        raise ValueError(f"No loader found for file extension: {ext}")

def load_documents(files: List["st.uploaded_file_manager.UploadedFile"]) -> List[Document]:
    documents = []
    for file in files:
        file_path = os.path.join("uploaded_files", file.name)
        os.makedirs("uploaded_files", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())

        document = load_single_document(file_path)
        documents.append(document)

    return documents

def get_answer(query: str):
    global llm

    if llm is None:
        qa = RetrievalQA(
            combine_documents_chain=[
                Chroma(persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
            ],
            retriever=Chroma(persist_directory=persist_directory, client_settings=CHROMA_SETTINGS),
            model=GPT4All,
            question_embedding_model=HuggingFaceEmbeddings(embeddings_model_name),
            vector_store=Chroma(persist_directory=persist_directory, client_settings=CHROMA_SETTINGS),
        )

        qa.load_model(
            model_type=model_type,
            model_path=model_path,
            model_n_ctx=int(model_n_ctx),
        )

        llm = qa.llm
        llm.load()

    answer = llm.answer(query)
    source_data = llm.get_data(query)

    return query, answer, source_data

def main():
    st.title("PrivateGPT - Language Model Demo")

    uploaded_files = st.file_uploader("Upload Document(s)", accept_multiple_files=True)
    query = st.text_input("Enter your question:")
    if st.button("Ask"):
        if query:
            documents = load_documents(uploaded_files) if uploaded_files else []
            for document in documents:
                st.markdown(f"**Uploaded Document:** {document}")
            question, answer, source_data = get_answer(query)
            st.markdown(f"**Question:** {question}")
            st.markdown(f"**Answer:** {answer}")
            if source_data:
                st.markdown("**Source Documents:**")
                for doc in source_data:
                    st.markdown(f"- {doc}")

if __name__ == "__main__":
    main()
