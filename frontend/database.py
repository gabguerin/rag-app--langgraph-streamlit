# pages/database.py
import os
from io import BytesIO
from pathlib import Path

import streamlit as st
from pypdf import PdfReader, PdfWriter
from streamlit_extras.stylable_container import stylable_container

from backend.vectorstore import MultiModalVectorstore

db = MultiModalVectorstore()


def show():
    st.title("Uploaded Files")

    # Ensure the directory exists
    p = Path("./database/documents")
    p.mkdir(parents=True, exist_ok=True)

    # List existing files
    for file_path in p.rglob("*"):
        nb_processed_pages, nb_pages = get_number_of_processed_pages(file_path)
        with stylable_container(
            key=f"file-{file_path.name}",
            css_styles="""{"border": "1px solid #ccc", "padding": "10px"}""",
        ):
            col1, col2, col3 = st.columns((4, 1, 1))
            col1.write(file_path.name)
            with col2:
                text_color = "green" if nb_processed_pages == nb_pages else "red"
                st.write(
                    f":{text_color}[{nb_processed_pages}/{nb_pages} pages processed]"
                )
            with col3:
                # Use a session state flag to trigger re-upload
                if st.button("Re-Upload", key=f"reupload-{file_path.name}"):
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
    nb_pages = len(PdfReader(pdf_filepath).pages)
    nb_processed_pages = 0
    for i in range(nb_pages):
        if not db.is_document_stored(pdf_filepath.name, i):
            break
        nb_processed_pages = i
    return nb_processed_pages, nb_pages
