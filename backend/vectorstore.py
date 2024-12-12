from io import BytesIO
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf

from backend.chat_models.llms import summarizer

# Constants for directory paths and embedding model
DOCUMENTS_PATH = Path("database/documents")
VECTORSTORE_PATH = "database/vectorstore"
OUTPUT_IMAGES_PATH = "database/output_images"

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIMENSION = 768


class MultiModalVectorstore:
    """
    A class for managing multi-modal vector storage with PDF processing, embedding,
    and retrieval functionalities.
    """

    def __init__(self):
        """Initialize the vector store and related configurations."""
        self._documents_path = DOCUMENTS_PATH
        self._vectorstore_path = VECTORSTORE_PATH

        # Initialize the embedding function and vectorstore
        self._embedding_function = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            dimensions=EMBEDDING_DIMENSION,
        )
        self._vectorstore = Chroma(
            collection_name="Oxfam-collection",
            persist_directory=self._vectorstore_path,
            embedding_function=self._embedding_function,
        )
        self._retriever = self._vectorstore.as_retriever(k=3)

    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve documents based on a query.

        Args:
            query (str): The search query.

        Returns:
            List[Document]: Retrieved documents.
        """
        return self._retriever.invoke(query)

    def get_all_documents_in_vectorstore(self) -> int:
        return self._vectorstore.get()["documents"]

    def get_nb_stored_pages_in_vectorstore(
        self, pdf_filename: str, page_number: int
    ) -> int:
        return len(
            self._vectorstore.get(
                where={
                    "filename": pdf_filename,
                },
            )["ids"]
        )

    def is_document_stored(self, pdf_filename: str, page_number: int) -> bool:
        """Check if a specific PDF page is already stored in the vectorstore.

        Args:
            pdf_filename (str): The name of the PDF file.
            page_number (int): The page number to check.

        Returns:
            bool: True if the page is already stored, False otherwise.
        """
        document_id = self._get_document_id(pdf_filename, page_number)
        return (
            len(
                self._vectorstore.get(
                    where={
                        "document_id": document_id,
                    },
                )["ids"]
            )
            > 0
        )

    def add_new_pdf_page_to_vectorstore(
        self, pdf_page: BytesIO, pdf_filename: str, page_number: int
    ) -> None:
        """Process and add a single PDF page to the vectorstore.

        Args:
            pdf_page (BytesIO): The PDF page content.
            pdf_filename (str): The name of the PDF file.
            page_number (int): The page number of the PDF.
        """
        # Partition the PDF into elements (text, tables, etc.)
        raw_pdf_elements = partition_pdf(
            file=pdf_page,
            extract_images_in_pdf=False,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
        )

        # Process elements and summarize tables if needed
        page_elements = [
            (
                self._summarize_table_element(element)
                if "documents.elements.Table" in str(type(element))
                else str(element)
            )
            for element in raw_pdf_elements
        ]

        # Create a Document object with metadata
        document_id = self._get_document_id(pdf_filename, page_number)
        page_document = Document(
            page_content="\n".join(page_elements),
            metadata={"document_id": document_id, "filename": pdf_filename},
        )

        print(
            f"Page content for {pdf_filename}, page {page_number}:\n\n{page_document}\n\n"
        )

        # Add the document to the vectorstore
        self._vectorstore.add_documents(documents=[page_document], ids=[document_id])

    def delete_file_from_vectorstore(self, pdf_filename: str) -> None:
        ids_to_delete = self._vectorstore.get(
            where={
                "filename": pdf_filename,
            },
        )["ids"]
        if len(ids_to_delete) > 0:
            return self._vectorstore.delete(
                ids=ids_to_delete,
            )
        return None

    @staticmethod
    def _summarize_table_element(table_element: Element) -> str:
        """Summarize a table element.

        Args:
            table_element (Element): The table element to summarize.

        Returns:
            str: The summarized content.
        """
        return summarizer.invoke(inputs={"element": table_element})

    @staticmethod
    def _get_document_id(pdf_filename: str, page_number: int) -> str:
        """Generate a unique document ID for a PDF page.

        Args:
            pdf_filename (str): The name of the PDF file.
            page_number (int): The page number.

        Returns:
            str: A unique document ID.
        """
        return f"{pdf_filename}::{page_number:05}"
