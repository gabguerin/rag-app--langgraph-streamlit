from pathlib import Path

import streamlit as st
from streamlit_extras.stylable_container import stylable_container

from backend.vectorstore import MultiModalVectorstore
from frontend.utils import (
    upload_file,
    get_number_of_processed_pages,
    delete_file_from_database,
)

db = MultiModalVectorstore()

BOOK_EMOJI_SHORTCODES = [
    ":closed_book:",
    ":green_book:",
    ":blue_book:",
    ":orange_book:",
]


def load_files_and_data():
    """Loads file names and related data into session state."""
    if "file_data" not in st.session_state:
        st.session_state.file_data = []
        p = Path("./database/documents")
        p.mkdir(parents=True, exist_ok=True)
        for file_path in p.rglob("*"):
            nb_processed_pages, nb_pages = get_number_of_processed_pages(db, file_path)
            st.session_state.file_data.append(
                {
                    "file_path": file_path,
                    "nb_processed_pages": nb_processed_pages,
                    "nb_pages": nb_pages,
                }
            )


def show():
    st.title(":books: Uploaded Files")

    # Load file data once
    load_files_and_data()

    with st.container(border=True, height=260):
        for idx, file_info in enumerate(st.session_state.file_data):
            file_path = file_info["file_path"]
            nb_processed_pages = file_info["nb_processed_pages"]
            nb_pages = file_info["nb_pages"]

            with stylable_container(
                key=f"file_{idx}",
                css_styles="""
                    {{
                        background-color: {color};
                    }}
                """.format(
                    color="#f9f9f9"
                ),
            ):
                col1, col2, col3 = st.columns((15, 1, 1), vertical_alignment="center")
                with col1:
                    col1.write(
                        f"{BOOK_EMOJI_SHORTCODES[idx % len(BOOK_EMOJI_SHORTCODES)]} {file_path.name}"
                    )
                with col2:
                    upload_completed = nb_processed_pages == nb_pages
                    text_color = "green" if upload_completed else "red"
                    with stylable_container(
                        key=f"reload_button_{idx}",
                        css_styles="""
                            button {{
                                float: right;
                                border: 0;
                                background-color: {color};
                            }}
                        """.format(
                            color="#f9f9f9"
                        ),
                    ):
                        if st.button(
                            (
                                ":material/download_done:"
                                if upload_completed
                                else ":material/upload:"
                            ),
                            help=(
                                "Upload completed: " if upload_completed else "Upload: "
                            )
                            + f":{text_color}[{nb_processed_pages}/{nb_pages} uploaded pages]",
                            key=f"upload-{file_path.name}",
                            disabled=upload_completed,
                        ):
                            st.session_state.file_reuploaded = file_path
                            st.rerun()

                with col3:
                    with stylable_container(
                        key=f"delete_button_{idx}",
                        css_styles="""
                            button {{
                                float: right;
                                border: 0;
                                background-color: {color};
                            }}
                        """.format(
                            color="#f9f9f9"
                        ),
                    ):
                        if st.button(
                            ":material/delete:",
                            help="Delete",
                            key=f"delete-{file_path.name}",
                        ):
                            delete_file_from_database(db, file_path)
                            st.session_state.file_data.pop(idx)
                            st.rerun()

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
            with pdf_filepath.open("wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.file_uploaded = pdf_filepath
            st.session_state.file_data.append(
                {
                    "file_path": pdf_filepath,
                    "nb_processed_pages": 0,
                    "nb_pages": 0,
                }
            )
            st.rerun()

    # Trigger file processing if needed
    if "file_uploaded" in st.session_state:
        upload_file(db, st.session_state["file_uploaded"])
        del st.session_state["file_uploaded"]
        st.rerun()

    if "file_reuploaded" in st.session_state:
        upload_file(db, st.session_state["file_reuploaded"])
        del st.session_state["file_reuploaded"]
        st.rerun()
