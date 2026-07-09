import os
import streamlit as st
from chatbot import create_vector_store, ask_question


# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Smart RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# Session State
# -------------------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:

    st.title("📂 Document Upload")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type="pdf"
    )

    if uploaded_file is not None:

        os.makedirs("documents", exist_ok=True)

        pdf_path = os.path.join(
            "documents",
            uploaded_file.name
        )

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing PDF..."):

            st.session_state.vector_db = create_vector_store(pdf_path)

        st.success("✅ PDF Uploaded Successfully!")

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("### 📖 About")
    st.write("""
This chatbot uses:

- 📄 PDF Loader
- 🧠 Embeddings
- 📦 FAISS Vector Database
- 🔍 Retrieval-Augmented Generation (RAG)
- 🤖 Google Gemini
""")

# -------------------------------
# Main Page
# -------------------------------

st.title("🤖 Smart RAG Chatbot")

st.caption("Ask questions about your uploaded document.")

if st.session_state.vector_db is None:
    st.info("👈 Upload a PDF from the sidebar to begin.")

# -------------------------------
# Display Chat History
# -------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------
# Chat Input
# -------------------------------

question = st.chat_input("Ask something about your document...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.vector_db is None:

        answer = "⚠ Please upload a PDF first."

    else:

        with st.spinner("Searching document..."):

            answer = ask_question(
                st.session_state.vector_db,
                question
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)