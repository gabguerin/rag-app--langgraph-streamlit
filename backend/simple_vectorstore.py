from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document

from langchain_nomic import NomicEmbeddings

DOCUMENTS_PATH = "database/documents"
VECTORSTORE_PATH = "database/vectorstore"


class VectorStore:

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, k: int = 3):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k

        self.documents_path = DOCUMENTS_PATH
        self.vectorstore_path = VECTORSTORE_PATH

        self.embedding_function = NomicEmbeddings(
            model="nomic-embed-text-v1.5", inference_mode="local"
        )
        self.vectorstore = Chroma(
            persist_directory=self.vectorstore_path,
            embedding_function=self.embedding_function,
        )

        self.update_vectorstore()
        self.retriever = self.vectorstore.as_retriever(k=self.k)

    def retrieve(self, query: str) -> list[Document]:
        if not self.retriever:
            raise ValueError("Retriever has not been initialized.")
        return self.retriever.invoke(query)

    def _load_documents(self) -> List[Document]:
        directory_loader = PyPDFDirectoryLoader(self.documents_path)
        return directory_loader.load()

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(documents)

    def update_vectorstore(self) -> None:
        documents = self._load_documents()
        chunks = self._split_documents(documents)

        # Assign unique IDs to chunks based on source and page information.
        chunks_with_ids = self._assign_chunk_ids(chunks)

        # Get existing document IDs in the database.
        existing_ids = set(self.vectorstore.get(include=[])["ids"])
        print(f"Vectorstore: Number of existing documents in DB: {len(existing_ids)}")

        # Filter out chunks that already exist in the database.
        new_chunks = [
            chunk
            for chunk in chunks_with_ids
            if chunk.metadata["id"] not in existing_ids
        ]

        if new_chunks:
            print(f"Vectorstore: Adding new documents: {len(new_chunks)}")
            self.vectorstore.add_documents(
                new_chunks, ids=[chunk.metadata["id"] for chunk in new_chunks]
            )
        else:
            print("Vectorstore: No new documents to add")

    @staticmethod
    def _assign_chunk_ids(chunks: list[Document]) -> list[Document]:
        """
        Assigns unique IDs to each chunk in the format 'source:page:chunk_index'.
        """
        last_page_id, current_chunk_index = None, 0

        for chunk in chunks:
            source, page = chunk.metadata.get("source"), chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # Reset chunk index if we're on a new page, else increment it.
            current_chunk_index = (
                current_chunk_index + 1 if current_page_id == last_page_id else 0
            )

            # Assign the computed ID to the chunk's metadata.
            chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

        return chunks
