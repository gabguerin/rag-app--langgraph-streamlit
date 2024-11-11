# pages/chat.py
import streamlit as st
from backend.rag_graph.graph import graph
from backend.vectorstore import PDFVectorstore


def show(db: PDFVectorstore):
    st.title("Oxfam Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Bonjour, comment puis-je vous aider ?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            inputs = {"question": prompt, "max_retries": 3}
            for event in graph.stream(inputs, stream_mode="values"):
                print(event)

            response = event["generation"]
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
