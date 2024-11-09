# pages/chat.py
import streamlit as st
from backend.rag_graph.graph import graph

def show():
    st.title("Oxfam Chatbot")
    dialogue_text = st.text_area(
        "Bonjour, comment puis-je vous aider ?",
        value="Quels sont les chiffres de TotalEnergies total en 2023 et sur le 4eme trimestre de 2023 ?",
        height=100
    )
    if st.button("Submit"):
        inputs = {
            "question": dialogue_text,
            "max_retries": 3
        }
        for event in graph.stream(inputs, stream_mode="values"):
            print(event)

        st.write(graph.get_state()["generation"], use_container_width=True)
