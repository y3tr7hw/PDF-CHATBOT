import os
import io
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import easyocr

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_chroma import Chroma

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")


def render_highlighted_page(filename, page_num, snippet, max_chars=200):
    pdf_bytes = st.session_state.pdf_bytes.get(filename)
    if not pdf_bytes:
        return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_num < 1 or page_num > len(doc):
        return None
    page = doc[page_num - 1]

    search_text = snippet[:max_chars].strip()
    instances = page.search_for(search_text) if search_text else []

    if not instances and len(search_text) > 40:
        instances = page.search_for(search_text[:40])

    for inst in instances:
        page.add_highlight_annot(inst)

    pix = page.get_pixmap(dpi=150)
    return pix.tobytes("png")


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

# Light custom styling for a ChatGPT-like feel
st.markdown("""
<style>
.block-container { max-width: 850px; padding-top: 2rem; }
.stChatMessage { padding: 0.75rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("📄 PDF Chatbot")

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = set()
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = {}
if "summary" not in st.session_state:
    st.session_state.summary = None
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "page_texts" not in st.session_state:
    st.session_state.page_texts = []
if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None

# -----------------------------
# Sidebar: Upload + Controls
# -----------------------------
with st.sidebar:
    st.header("Setup")

    accent_options = {
        "US English": "com",
        "UK English": "co.uk",
        "Indian English": "co.in",
        "Australian English": "com.au",
    }
    accent_choice = st.selectbox("🔊 Voice accent", list(accent_options.keys()))
    st.session_state.voice_accent = accent_options[accent_choice]

    uploaded_files = st.file_uploader(
        "Upload your PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    current_names = {f.name for f in uploaded_files} if uploaded_files else set()
    if not current_names and st.session_state.processed_file_names:
        st.session_state.vector_db = None
        st.session_state.processed_file_names = set()
        st.session_state.messages = []
        st.session_state.summary = None
        st.session_state.suggested_questions = []
        st.session_state.page_texts = []
        st.session_state.pdf_bytes = {}

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.processed_file_names:
        st.success(f"Loaded: {', '.join(st.session_state.processed_file_names)}")

    if st.session_state.page_texts:
        st.divider()
        st.subheader("🔍 Search in PDF")
        search_term = st.text_input("Find text", key="pdf_search")
        if search_term:
            matches = [
                (fname, page_num, text) for fname, page_num, text in st.session_state.page_texts
                if search_term.lower() in text.lower()
            ]
            if matches:
                st.caption(f"Found on {len(matches)} page(s)")
                for fname, page_num, text in matches[:10]:
                    idx = text.lower().find(search_term.lower())
                    start = max(0, idx - 60)
                    end = min(len(text), idx + len(search_term) + 60)
                    snippet = text[start:end].replace(search_term, f"**{search_term}**")
                    st.markdown(f"**{fname} — Page {page_num}:** ...{snippet}...")
            else:
                st.caption("No matches found.")

    if st.session_state.messages:
        st.divider()
        st.subheader("💾 Export Chat")

        def build_export_text():
            names = ", ".join(st.session_state.processed_file_names) or "Untitled"
            lines = [f"# Chat Export — {names}", ""]
            for m in st.session_state.messages:
                role = "You" if m["role"] == "user" else "Assistant"
                lines.append(f"**{role}:** {m['content']}")
                lines.append("")
            return "\n".join(lines)

        export_text = build_export_text()

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Markdown",
                data=export_text,
                file_name="chat_export.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "⬇️ TXT",
                data=export_text.replace("**", ""),
                file_name="chat_export.txt",
                mime="text/plain",
                use_container_width=True
            )

# -----------------------------
# Process PDF (only once per file)
# -----------------------------
new_files = [f for f in (uploaded_files or []) if f.name not in st.session_state.processed_file_names]

if new_files:
    with st.status("Processing PDF(s)...", expanded=True) as status:

        all_page_texts = list(st.session_state.page_texts)
        all_chunks = []
        all_metadatas = []

        for uploaded_file in new_files:
            status.write(f"Reading {uploaded_file.name}...")
            st.session_state.pdf_bytes[uploaded_file.name] = uploaded_file.getvalue()

            reader = PdfReader(uploaded_file)
            file_page_texts = []
            for i, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted and extracted.strip():
                    file_page_texts.append((uploaded_file.name, i + 1, extracted))

            if not file_page_texts:
                status.write(f"Scanned PDF detected in {uploaded_file.name}. Running OCR...")
                reader_ocr = easyocr.Reader(['en'])
                images = convert_from_bytes(uploaded_file.getvalue())
                for i, image in enumerate(images):
                    result = reader_ocr.readtext(np.array(image))
                    page_text = "\n".join(item[1] for item in result)
                    if page_text.strip():
                        file_page_texts.append((uploaded_file.name, i + 1, page_text))

            all_page_texts.extend(file_page_texts)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            for fname, page_num, page_text in file_page_texts:
                for c in splitter.split_text(page_text):
                    all_chunks.append(c)
                    all_metadatas.append({"source": fname, "page": page_num})

            st.session_state.processed_file_names.add(uploaded_file.name)

        st.session_state.page_texts = all_page_texts

        status.write("Building vector database...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        st.session_state.vector_db = Chroma.from_texts(
            texts=all_chunks,
            embedding=embeddings,
            metadatas=all_metadatas
        )

        status.write("Generating summary and suggested questions...")
        llm = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", google_api_key=GOOGLE_API_KEY)
        preview_text = "\n\n".join(all_chunks[:8])

        summary_resp = llm.invoke(f"Summarize this document set in 4-6 concise sentences:\n\n{preview_text}")
        st.session_state.summary = (
            summary_resp.content if isinstance(summary_resp.content, str)
            else "".join(b.get("text", "") for b in summary_resp.content if isinstance(b, dict))
        )

        q_resp = llm.invoke(
            f"List exactly 4 short specific questions a reader might ask about this content. "
            f"One per line, no numbering:\n\n{preview_text}"
        )
        q_text = (
            q_resp.content if isinstance(q_resp.content, str)
            else "".join(b.get("text", "") for b in q_resp.content if isinstance(b, dict))
        )
        st.session_state.suggested_questions = [
            line.strip("-•0123456789. ").strip() for line in q_text.split("\n") if line.strip()
        ][:4]

        st.session_state.messages = []
        status.update(label="✅ Ready to chat!", state="complete", expanded=False)

# -----------------------------
# Chat Interface
# -----------------------------
if st.session_state.vector_db is not None:

    vector_db = st.session_state.vector_db
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    if st.session_state.summary:
        with st.expander("📋 Document Summary", expanded=True):
            st.markdown(st.session_state.summary)

    if st.session_state.suggested_questions:
        st.write("**Suggested questions:**")
        cols = st.columns(len(st.session_state.suggested_questions))
        for col, q in zip(cols, st.session_state.suggested_questions):
            if col.button(q, use_container_width=True):
                st.session_state.pending_question = q

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest",
        google_api_key=GOOGLE_API_KEY
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    col_input, col_mic = st.columns([6, 1])

    with col_mic:
        audio_value = st.audio_input("🎤", label_visibility="collapsed")

    with col_input:
        question = st.chat_input("Ask a question about your PDF")

    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
        if st.session_state.get("last_audio_bytes") != audio_bytes:
            st.session_state.last_audio_bytes = audio_bytes
            recognizer = sr.Recognizer()
            with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                audio_data = recognizer.record(source)
            try:
                transcribed = recognizer.recognize_google(audio_data)
                st.session_state.pending_question = transcribed
                st.rerun()
            except sr.UnknownValueError:
                st.error("Could not understand the audio. Try recording again.")
            except sr.RequestError as e:
                st.error(f"Speech recognition service error: {e}")

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        docs = retriever.invoke(question)
        context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', '?')}, Page {doc.metadata.get('page', '?')}] {doc.page_content}"
            for doc in docs
        )
        sources_cited = sorted(
            {(doc.metadata.get("source", "?"), doc.metadata.get("page", 0)) for doc in docs},
            key=lambda s: (s[0], s[1])
        )

        history_text = ""
        for m in st.session_state.messages[-7:-1]:
            role = "User" if m["role"] == "user" else "Assistant"
            history_text += f"{role}: {m['content']}\n"

        prompt = f"""You are answering questions about a PDF document.
Use ONLY the context below to answer. If the answer isn't in the context, say so.
Each context snippet is tagged with its source file and page — refer to them when useful.

Conversation so far:
{history_text}

Context:
{context}

Question:
{question}
"""

        with st.chat_message("assistant"):
            def stream_response():
                for chunk in llm.stream(prompt):
                    content = chunk.content
                    if isinstance(content, str):
                        if content:
                            yield content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                yield block.get("text", "")

            answer = st.write_stream(stream_response)

            try:
                tts = gTTS(text=answer, lang='en', tld=st.session_state.get("voice_accent", "com"))
                audio_fp = io.BytesIO()
                tts.write_to_fp(audio_fp)
                audio_fp.seek(0)
                st.audio(audio_fp, format="audio/mp3")
            except Exception as e:
                st.caption(f"🔇 Voice output unavailable: {e}")

            if sources_cited:
                caption = " | ".join(f"{src} (p.{pg})" for src, pg in sources_cited)
                st.caption(f"📍 Source: {caption}")

                for src, pg in sources_cited:
                    matching_docs = [
                        d.page_content for d in docs
                        if d.metadata.get("source") == src and d.metadata.get("page") == pg
                    ]
                    snippet = matching_docs[0] if matching_docs else ""
                    with st.expander(f"🔍 View highlighted source — {src}, page {pg}"):
                        img_bytes = render_highlighted_page(src, pg, snippet)
                        if img_bytes:
                            st.image(img_bytes, use_container_width=True)
                        else:
                            st.caption("Could not render this page.")

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Upload a PDF from the sidebar to get started.")