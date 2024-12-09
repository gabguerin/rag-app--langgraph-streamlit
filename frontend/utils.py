import os
from io import BytesIO
from pathlib import Path
from typing import Tuple

from pypdf import PdfReader, PdfWriter
import streamlit as st

from backend.vectorstore import MultiModalVectorstore


def upload_file(db: MultiModalVectorstore, pdf_filepath: Path):
    pdf_file = PdfReader(pdf_filepath)
    total_pages = len(pdf_file.pages)

    progress_placeholder = st.progress(0, text=f"Processing {pdf_filepath.name}")
    for i, page in enumerate(pdf_file.pages):
        if not db.is_document_stored(pdf_filename=pdf_filepath.name, page_number=i):
            print(f"PROCESSING {pdf_filepath.name}: PAGE #{i+1}")
            writer = PdfWriter()
            writer.add_page(page)

            # Write the single page to a BytesIO object
            pdf_bytes = BytesIO()
            writer.write(pdf_bytes)
            pdf_bytes.seek(0)

            # Add to vector store
            db.add_new_pdf_page_to_vectorstore(
                pdf_page=pdf_bytes, pdf_filename=pdf_filepath.name, page_number=i
            )
        # Update the progress bar
        progress_placeholder.progress(
            (i + 1) / total_pages, text=f"Processing page #{i+2} of {pdf_filepath.name}"
        )

    progress_placeholder.empty()


def get_number_of_processed_pages(
    db: MultiModalVectorstore, pdf_filepath: Path
) -> Tuple:
    """Returns the number of processed and total pages in a PDF file.

    Args:
        db: MultiModalVectorstore object
        pdf_filepath (Path): Path to the PDF file.

    Returns:
        tuple: (number of processed pages, total number of pages)
    """
    pdf_reader = PdfReader(pdf_filepath)
    total_pages = len(pdf_reader.pages)

    processed_pages = next(
        (
            i
            for i in range(total_pages)
            if not db.is_document_stored(pdf_filepath.name, i)
        ),
        total_pages,
    )

    return processed_pages, total_pages


def delete_file_from_database(db: MultiModalVectorstore, pdf_filepath: Path) -> None:
    """Deletes the pdf file from local storage and from vectorstore

    Args:
        db: MultiModalVectorstore object
        pdf_filepath (Path): Path to the PDF file.

    """
    print("Remove documents from vectorstore")
    db.delete_file_from_vectorstore(pdf_filename=pdf_filepath.name)
    print("Remove file from database")
    pdf_filepath.unlink()
    print(f"{pdf_filepath.name} deleted")
