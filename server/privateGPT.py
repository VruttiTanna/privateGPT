import streamlit as st
import os
from langchain.docstore.document import Document
from langchain import DocumentLoader
from gpt4all import GPT, set_random_seed

# Set random seed for reproducibility
set_random_seed(42)

# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = int(os.environ.get('MODEL_N_CTX'))

# Initialize GPT model
gpt = GPT(model_path, model_n_ctx)

# Initialize document loader
document_loader = DocumentLoader()

# Streamlit App
st.title("PrivateGPT - Question Answering System")

# File Upload
uploaded_files = st.file_uploader("Upload document(s)", accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        document = document_loader.load_file(uploaded_file)
        documents.append(document)

    st.write(f"Loaded {len(documents)} document(s)")
    st.write("Document(s):")
    for document in documents:
        st.write(document.metadata["filename"])

    # Process the document(s)
    for document in documents:
        # Process the document
        # ...

        # Perform question answering
        question = st.text_input("Enter your question")
        if question:
            # Perform question answering using GPT
            answer = gpt.ask(question, document.content)

            # Display the answer
            st.write("Question:", question)
            st.write("Answer:", answer)
