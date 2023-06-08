import streamlit as st
import os
import glob
from typing import List
from langchain.docstore.document import Document
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

# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = int(os.environ.get('MODEL_N_CTX'))

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

class CustomTextLoader:
    @staticmethod
    def load(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return [Document(content=content, metadata={})]

def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]
    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from the source documents directory
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True))
    return [load_single_document(file_path) for file_path in all_files]

# Streamlit App
st.title("PrivateGPT - Question Answering System")

# File Upload
uploaded_file = st.file_uploader("Upload a document", type=list(LOADER_MAPPING.keys()))

if uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")
    st.write("Uploaded file contents:")
    st.code(file_contents)

    # Process the uploaded document
    document = Document(content=file_contents, metadata={})
    texts = [document]

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # Create and store the vector store
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    db.persist()

    # Ask a question
    question = st.text_input("Ask a question")
    if st.button("Get Answer"):
        if question.strip() != "":
            # Get the answer using the loaded model
            answer, source_data = get_answer(question, db)
            st.write("Answer:", answer)
            st.write("Source Documents:")
            for source in source_data:
                st.write("- ", source.get("name"))
        else:
            st.warning("Please enter a question.")
else:
    st.warning("Please upload a document.")

