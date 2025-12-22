

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()


class LLMFactory:

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self._llm = ChatOpenAI(model=model_name)

    def get_instance(self):
        return self._llm


class VectorStoreFactory:

    def __init__(self, api_key, index_name: str = "tax-index-large"):
        self._pc = Pinecone(api_key=api_key)
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self._database = PineconeVectorStore(
            embedding=self._embeddings,
            index=self._pc.Index(index_name),
        )

    def get_instance(self):
        return self._database

    def as_retriever(self):
        return self._database.as_retriever(search_kwargs={"k": 4})
