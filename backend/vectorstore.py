import uuid
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_nomic import NomicEmbeddings
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf

from backend.chat_models.llms import summarizer

DOCUMENTS_PATH = Path("database/documents")
VECTORSTORE_PATH = "database/vectorstore"
TABLE_VECTORSTORE_PATH = "database/table_vectorstore"
OUTPUT_IMAGES_PATH = "database/output_images"

EMBEDDING_MODEL_NAME = "nomic-embed-text-v1.5"


class MultiModalVectorstore:

    def __init__(self):
        self.documents_path = DOCUMENTS_PATH
        self.vectorstore_path = VECTORSTORE_PATH
        self.table_vectorstore_path = TABLE_VECTORSTORE_PATH

        self.embedding_function = NomicEmbeddings(
            model=EMBEDDING_MODEL_NAME, inference_mode="local"
        )
        self.vectorstore = Chroma(
            persist_directory=self.vectorstore_path,
            embedding_function=self.embedding_function,
        )
        self.retriever = self.vectorstore.as_retriever(k=3)

        self.table_vectorstore = Chroma(
            persist_directory=self.table_vectorstore_path,
            embedding_function=self.embedding_function,
        )
        self.table_retriever = self.table_vectorstore.as_retriever(k=3)

    def retrieve(self, query: str) -> list[Document]:
        return self.table_retriever.invoke(query) + self.retriever.invoke(query)

    def is_document_stored(self, pdf_filename: str, page_number: int) -> bool:
        """Check if a PDF document is already stored in the document store by filename."""
        stored_docs = self.vectorstore.similarity_search(
            "", filter={"page_id": f"{pdf_filename}-{page_number}"}
        )
        return len(stored_docs) > 0

    def add_new_pdf_page_to_vectorstore(
        self, pdf_page: BytesIO, pdf_filename: str, page_number: int
    ):
        raw_pdf_elements = partition_pdf(
            file=pdf_page,
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
            print(str(element))
            if "documents.elements.Table" in str(type(element)):
                table_elements.append(str(element))
            elif "documents.elements.CompositeElement" in str(type(element)):
                text_elements.append(str(element))

        text_ids = [str(uuid.uuid4()) for _ in text_elements]
        text_chunks = [
            Document(
                page_content=element,
                metadata={
                    "document_id": text_ids[idx],
                    "filename": pdf_filename,
                    "page_id": f"{pdf_filename}-{page_number}",
                    "type": "text",
                },
            )
            for idx, element in enumerate(text_elements)
        ]
        self.vectorstore.add_documents(text_chunks, ids=text_ids)

        table_ids, summarized_table_elements = self._summarize_table_elements(
            table_elements, pdf_filename=pdf_filename, page_number=page_number
        )
        self.table_vectorstore.add_documents(summarized_table_elements, ids=table_ids)

    @staticmethod
    def _summarize_table_elements(
        table_elements: List[str], pdf_filename: str, page_number: int
    ) -> Tuple[List[str], List[Document]]:
        summarized_tables = [
            summarizer.invoke(inputs={"element": element})
            for element in tqdm(
                table_elements,
                desc=f"Summarizing table elements of {pdf_filename}...",
            )
        ]

        document_ids = [str(uuid.uuid4()) for _ in table_elements]
        return document_ids, [
            Document(
                page_content=element,
                metadata={
                    "document_id": document_ids[idx],
                    "filename": pdf_filename,
                    "page_id": f"{pdf_filename}-{page_number}",
                    "type": "table",
                },
            )
            for idx, element in enumerate(summarized_tables)
        ]
