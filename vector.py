import os

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

loader = Docx2txtLoader(
    "/Users/ksj/MyProjects/llm/inflearn-streamlit/tax_with_markdown.docx"
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
documents = loader.load_and_split(text_splitter)

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = "tax-index-large"
pc = Pinecone(api_key=pinecone_api_key)

vector_stroe = PineconeVectorStore(
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"), 
    index=pc.Index(pinecone_index_name)
)

vector_stroe.add_documents(documents)
print("Vector store created and documents added.")
