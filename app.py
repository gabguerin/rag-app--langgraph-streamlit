import streamlit as st
from dotenv import load_dotenv

from frontend import chat, database

# Load env variables
load_dotenv()

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
