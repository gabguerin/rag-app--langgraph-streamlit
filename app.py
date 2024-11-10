import os

import streamlit as st
from frontend import chat, database


os.environ["TAVILY_API_KEY"] = "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"
# _set_env("TAVILY_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
# _set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "local-llama32-rag"

st.set_page_config(page_title="Oxfam RAG App", layout="centered")

# Initialize session state for page selection
if "page" not in st.session_state:
    st.session_state.page = "Chat"  # Default page is "Chat"

# Sidebar for navigation
with st.sidebar:
    if st.button("Chat", icon=":material/smart_toy:", use_container_width=True):
        st.session_state.page = "Chat"
    if st.button("Database", icon=":material/database:", use_container_width=True):
        st.session_state.page = "Database"

# Render the selected page based on session state
if st.session_state.page == "Chat":
    chat.show()
elif st.session_state.page == "Database":
    database.show()
