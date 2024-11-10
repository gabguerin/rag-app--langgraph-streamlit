# pages/database.py
import time
from pathlib import Path

import pandas as pd
import streamlit as st

def show():
    st.title("Upload Files")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.write("Uploaded Files:")
        for uploaded_file in uploaded_files:
            st.write(f"- {uploaded_file.name}")

    st.dataframe(file_list_in_directory(), use_container_width=True, hide_index=True)

def file_list_in_directory():
    p = Path("./database/documents")
    all_files = []
    for i in p.rglob('*'):
        all_files.append((time.ctime(i.stat().st_ctime), i.name))
    columns = ["Created", "File name"]
    return pd.DataFrame.from_records(all_files, columns=columns)
