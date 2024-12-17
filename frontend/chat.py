import streamlit as st
from backend.rag_graph.graph import graph


def show():
    st.title(":robot_face: Oxfam Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Bonjour, posez-moi n'importe-quelle question sur une entreprise du SBF120 j'essaierai d'y répondre au mieux.",
            }
        ]

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Bonjour, comment puis-je vous aider ?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            response = "Désolé, je n'ai pas compris votre question."  # Default fallback
            try:
                inputs = {
                    "question": prompt,
                    "history": st.session_state.messages,
                    "max_retries": 3,
                }
                response_parts = []
                # Stream the response for dynamic updates
                for event in graph.stream(inputs, stream_mode="values"):
                    chunk = event.get("generation", "")
                    response_parts.append(chunk)
                    st.markdown("".join(response_parts))  # Update progressively
                response = "".join(response_parts)
            except Exception as e:
                st.error("Une erreur est survenue. Veuillez réessayer plus tard.")
                response = f"Erreur : {str(e)}"

            # Display final response
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
