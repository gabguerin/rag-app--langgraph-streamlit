from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings

URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


class Retriever:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, k: int = 3):
        self.urls = URLS
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.vectorstore = None
        self.retriever = None
        self._initialize_retriever()

    def _initialize_retriever(self):
        # Load documents
        documents = [WebBaseLoader(url).load() for url in self.urls]
        documents_list = [item for sublist in documents for item in sublist]

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        doc_splits = text_splitter.split_documents(documents_list)

        # Add to vectorDB
        self.vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
        )

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(k=self.k)

    def invoke(self, query: str):
        if not self.retriever:
            raise ValueError("Retriever has not been initialized.")
        return self.retriever.invoke(query)
