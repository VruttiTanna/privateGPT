import streamlit as st
import requests
import os
import glob
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
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()

# Set Streamlit configuration
st.set_page_config(page_title="LangChain Demo")

# Define the Chroma settings
CHROMA_SETTINGS = {
    "chroma_db_impl": "duckdb+parquet",
    "persist_directory": "db/",
    "anonymized_telemetry": False,
}

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

def upload_doc(document):
    if document is not None:
        filename = document.name
        save_path = os.path.join("source_documents", filename)
        with open(save_path, "wb") as f:
            f.write(document.getbuffer())
        return "Document upload successful"
    return "No selected file", 400

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
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    return [load_single_document(file_path) for file_path in all_files]

def ingest_data():
    # Load environment variables
    persist_directory = os.environ.get("PERSIST_DIRECTORY")
    source_directory = os.environ.get("SOURCE_DIRECTORY", "source_documents")
    embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")

    # Load documents and split into chunks
    st.write(f"Loading documents from {source_directory}")
    chunk_size = 500
    chunk_overlap = 50
    documents = load_documents(source_directory)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    st.write(f"Loaded {len(documents)} documents from {source_directory}")
    st.write(f"Split into {len(texts)} chunks of text (max. {chunk_size} characters each)")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # Create and store locally vectorstore
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None
    st.success("Ingestion completed successfully.")

def get_answer(query):
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    if llm is None:
        return "Model not downloaded", 400
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    if query is not None and query != "":
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        source_data = []
        for document in docs:
            source_data.append({"name": document.metadata["source"]})

        return query, answer, source_data

    return "Empty Query", 400

def upload_doc(document):
    if document.filename == "":
        return "No selected file", 400

    filename = document.filename
    save_path = os.path.join("source_documents", filename)
    document.save(save_path)

    return "Document upload successful"

def download_and_save():
    url = "https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin"  # Specify the URL of the resource to download
    filename = "ggml-gpt4all-j-v1.3-groovy.bin"  # Specify the name for the downloaded file
    models_folder = "models"  # Specify the name of the folder inside the Flask app root

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    bytes_downloaded = 0
    file_path = f"{models_folder}/{filename}"

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                bytes_downloaded += len(chunk)

                # Display download progress
                progress = bytes_downloaded / total_size
                st.progress(progress)

    return file_path

def main():
    global llm
    st.title("LangChain Demo")
    st.subheader("Upload a Document")
    document = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
    if st.button("Upload"):
        upload_doc(document)

    st.subheader("Download GPT4All Model")
    if st.button("Download"):
        st.text("Downloading...")
        file_path = download_and_save()
        st.success(f"Model downloaded and saved at {file_path}")
        llm = GPT4All(model_path=file_path, model_type=model_type, model_n_ctx=model_n_ctx)
        st.success("GPT4All model loaded successfully.")

    st.subheader("Ingest Data")
    if st.button("Ingest"):
        ingest_data()

    st.subheader("Ask a Question")
    query = st.text_input("Enter your question")
    if st.button("Submit"):
        question, answer, source_data = get_answer(query)
        st.write(f"Question: {question}")
        st.write(f"Answer: {answer}")

        if source_data:
            st.write("Source Documents:")
            for document in source_data:
                st.write(f"- {document['name']}")

if __name__ == "__main__":
    main()
