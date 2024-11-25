# pages/database.py
from io import BytesIO
from pathlib import Path

import streamlit as st
from pypdf import PdfReader, PdfWriter

from backend.vectorstore import MultiModalVectorstore

db = MultiModalVectorstore()

BOOK_EMOJI_SHORTCODES = [
    ":closed_book:",
    ":green_book:",
    ":blue_book:",
    ":orange_book:",
]


def show():
    st.title(":books: Uploaded Files")

    # Ensure the directory exists
    p = Path("./database/documents")
    p.mkdir(parents=True, exist_ok=True)

    # List existing files
    for idx, file_path in enumerate(p.rglob("*")):
        nb_processed_pages, nb_pages = get_number_of_processed_pages(file_path)
        with st.container(border=True):
            col1, col2 = st.columns((4, 1), vertical_alignment="center")
            with col1:
                st.write(
                    f"{BOOK_EMOJI_SHORTCODES[idx%len(BOOK_EMOJI_SHORTCODES)]} {file_path.name}"
                )
                text_color = "green" if nb_processed_pages == nb_pages else "red"
                st.caption(
                    f":{text_color}[{nb_processed_pages}/{nb_pages} uploaded pages]"
                )
            with col2:
                disable_button = nb_processed_pages == nb_pages
                # Use a session state flag to trigger re-upload
                if st.button(
                    "Re-Upload",
                    key=f"reupload-{file_path.name}",
                    disabled=disable_button,
                ):
                    st.session_state["file_reuploaded"] = file_path
                    st.rerun()  # Rerun app to trigger reupload

    # Handle uploaded file
    uploaded_file = st.file_uploader(
        "Choose files",
        type=["pdf", "txt"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded_file:
        pdf_filepath = Path("database/documents") / uploaded_file.name
        if not pdf_filepath.exists():
            # Save uploaded file
            with pdf_filepath.open("wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state["file_uploaded"] = pdf_filepath
            st.rerun()  # Rerun app after upload

    # Trigger file processing if needed
    if "file_uploaded" in st.session_state:
        upload_file(st.session_state["file_uploaded"])
        del st.session_state["file_uploaded"]

    if "file_reuploaded" in st.session_state:
        upload_file(st.session_state["file_reuploaded"])
        del st.session_state["file_reuploaded"]


def upload_file(pdf_filepath: Path):
    pdf_file = PdfReader(pdf_filepath)
    total_pages = len(pdf_file.pages)

    progress_placeholder = st.progress(0, text=f"Processing {pdf_filepath.name}")
    for i, page in enumerate(pdf_file.pages):
        if not db.is_document_stored(pdf_filename=pdf_filepath.name, page_number=i):
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
            (i + 1) / total_pages, text=f"Processing {pdf_filepath.name}"
        )

    progress_placeholder.empty()


def get_number_of_processed_pages(pdf_filepath: Path):
    """Returns the number of processed and total pages in a PDF file.

    Args:
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
