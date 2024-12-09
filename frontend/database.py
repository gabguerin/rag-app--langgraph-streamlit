# pages/database.py
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


def show():
    st.title(":books: Uploaded Files")

    # Ensure the directory exists
    p = Path("./database/documents")
    p.mkdir(parents=True, exist_ok=True)

    with st.container(border=True, height=260):
        # List existing files
        for idx, file_path in enumerate(p.rglob("*")):

            nb_processed_pages, nb_pages = get_number_of_processed_pages(db, file_path)
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
                col1, col2, col3 = st.columns((14, 1, 1), vertical_alignment="center")
                with col1:
                    col11, col12 = st.columns((4, 1), vertical_alignment="center")
                    col11.write(
                        f"{BOOK_EMOJI_SHORTCODES[idx%len(BOOK_EMOJI_SHORTCODES)]} {file_path.name}"
                    )
                    text_color = "green" if nb_processed_pages == nb_pages else "red"
                    col12.caption(
                        f":{text_color}[{nb_processed_pages}/{nb_pages} uploaded pages]"
                    )
                with col2:
                    disable_button = nb_processed_pages == nb_pages
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
                        # Use a session state flag to trigger re-upload
                        if st.button(
                            ":arrow_forward:",
                            help="Resume",
                            key=f"resume-{file_path.name}",
                            disabled=disable_button,
                        ):
                            st.session_state["file_reuploaded"] = file_path
                            st.rerun()

                with col3:
                    with stylable_container(
                        key=f"remove_button_{idx}",
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
                            ":x:",
                            help="Delete",
                            key=f"delete-{file_path.name}",
                        ):
                            delete_file_from_database(db, file_path)
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
            # Save uploaded file
            with pdf_filepath.open("wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state["file_uploaded"] = pdf_filepath
            st.rerun()  # Rerun app after upload

    # Trigger file processing if needed
    if "file_uploaded" in st.session_state:
        upload_file(db, st.session_state["file_uploaded"])
        del st.session_state["file_uploaded"]
        st.rerun()

    if "file_reuploaded" in st.session_state:
        upload_file(db, st.session_state["file_reuploaded"])
        del st.session_state["file_reuploaded"]
        st.rerun()
