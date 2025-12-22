import os

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from factory import VectorStoreFactory

load_dotenv()


def format_documents(documents):
    return "\n\n".join(d.page_content for d in documents)


def load_documents_from_docx(file_path):    
    loader = Docx2txtLoader(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    return loader.load_and_split(text_splitter)


# documents = load_documents_from_docx("/Users/ksj/MyProjects/llm/inflearn-streamlit/tax_with_markdown.docx")

# print(f"Loaded {len(documents)} documents from the DOCX file.")

# database = VectorStoreFactory(os.environ.get("PINECONE_API_KEY")).get_instance()
# database.add_documents(documents)

# print("Vector store created and documents added.")
