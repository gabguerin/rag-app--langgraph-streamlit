import uuid
from pathlib import Path
from typing import List, Tuple

from langchain.retrievers import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_nomic import NomicEmbeddings
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf

from backend.chat_models import TableNTextSummarizer

DOCUMENTS_PATH = Path("database/documents")
VECTORSTORE_PATH = "database/vectorstore"
OUTPUT_IMAGES_PATH = "database/output_images"

EMBEDDING_MODEL_NAME = "nomic-embed-text-v1.5"


class PDFVectorstore:

    def __init__(self):

        self.documents_path = DOCUMENTS_PATH
        self.vectorstore_path = VECTORSTORE_PATH

        self.embedding_function = NomicEmbeddings(
            model=EMBEDDING_MODEL_NAME, inference_mode="local"
        )
        self.vectorstore = Chroma(
            collection_name="summaries",
            persist_directory=self.vectorstore_path,
            embedding_function=self.embedding_function,
        )

        # The storage layer for the parent documents
        # self.document_store = InMemoryStore()
        # self.document_store_id_key = "document_id"

        # self.retriever = MultiVectorRetriever(
        #     vectorstore=self.vectorstore,
        #     docstore=self.document_store,
        #     id_key=self.document_store_id_key,
        # )

        self.update_vectorstore()

        self.retriever = self.vectorstore.as_retriever(k=3)

    def retrieve(self, query: str) -> list[Document]:
        if not self.retriever:
            raise ValueError("Retriever has not been initialized.")
        return self.retriever.invoke(query)

    def update_vectorstore(self):
        for pdf_filepath in self.documents_path.rglob("*.pdf"):
            pdf_filename = pdf_filepath.name
            print("Processing " + pdf_filename)
            # Check if the document has already been processed
            if not self._is_document_stored(pdf_filename):
                self._add_new_file_to_vectorstore(pdf_filename)

    def _is_document_stored(self, pdf_filename: str) -> bool:
        """Check if a PDF document is already stored in the document store by filename."""
        stored_docs = self.vectorstore.get(where={"filename": pdf_filename})
        print(stored_docs)
        return len(stored_docs["documents"]) > 0

    def _add_new_file_to_vectorstore(self, pdf_filename: str):
        raw_pdf_elements = partition_pdf(
            filename=self.documents_path / pdf_filename,
            extract_images_in_pdf=False,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path=OUTPUT_IMAGES_PATH,
        )

        table_elements, text_elements = [], []
        for element in raw_pdf_elements:
            str(element)
            if "unstructured.documents.elements.Table" in str(type(element)):
                table_elements.append(str(element))
            elif "unstructured.documents.elements.CompositeElement" in str(
                type(element)
            ):
                text_elements.append(str(element))

        print("Summarizing table chunks in " + pdf_filename)
        table_ids, table_chunks = self._summarize_table_or_text_elements(
            table_elements, pdf_filename, type="table"
        )

        print("Summarizing text in " + pdf_filename)
        text_ids, text_chunks = self._summarize_table_or_text_elements(
            text_elements, pdf_filename, type="text"
        )

        self.vectorstore.add_documents(table_chunks + text_chunks)

        # self.retriever.docstore.mset(
        #     list(zip(table_ids, table_chunks)) + list(zip(text_ids, text_chunks))
        # )
        # self.retriever.vectorstore.add_documents(table_chunks + text_chunks)

    @staticmethod
    def _summarize_table_or_text_elements(
        table_or_text_elements: List[str], pdf_filename: str, type: str
    ) -> Tuple[List[str], List[Document]]:
        summarizer = TableNTextSummarizer()
        summarized_elements = [
            summarizer.invoke(element=element)
            for element in tqdm(
                table_or_text_elements, desc=f"Summarizing {type} elements..."
            )
        ]

        document_ids = [str(uuid.uuid4()) for _ in table_or_text_elements]
        return document_ids, [
            Document(
                page_content=element,
                metadata={
                    "document_id": document_ids[idx],
                    "filename": pdf_filename,
                    "type": type,
                },
            )
            for idx, element in enumerate(summarized_elements)
        ]
