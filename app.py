import streamlit as st

st.set_page_config(page_title="AI Hub", page_icon="🤖", layout="wide")

st.title("🤖 Welcome to Your AI Hub")
st.write("Choose a mode from the sidebar to get started:")

col1, col2 = st.columns(2)
with col1:
    st.subheader("💬 General Chat")
    st.write("Chat freely with EDITH — choose a personality, model, and creativity level.")
with col2:
    st.subheader("📄 PDF Chat")
    st.write("Upload PDFs and ask questions with citations, search, voice, and highlighting.")

st.info("👈 Use the sidebar to switch between modes.")