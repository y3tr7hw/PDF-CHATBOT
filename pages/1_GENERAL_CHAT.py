import os
import json
import uuid
import streamlit as st
from dotenv import load_dotenv
from google import genai
import io
import speech_recognition as sr
from gtts import gTTS

# ----------------------------
# Load API Key
# ----------------------------
load_dotenv()
os.makedirs("chats", exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

def get_chat_title(messages):
    for msg in messages:
        if msg["role"] == "user":
            title = msg["content"][:30]
            if len(msg["content"]) > 30:
                title += "..."
            return title
    return "New Chat"

if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="EDITH",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("🤖 EDITH")

    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_id = str(uuid.uuid4())
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("💬 Saved Chats")

    chat_files = os.listdir("chats")
    for chat in chat_files:
        with open(f"chats/{chat}", "r") as file:
            messages = json.load(file)
        title = get_chat_title(messages)
        if st.button(title, key=chat):
            st.session_state.messages = messages
            st.session_state.chat_id = chat.replace(".json", "")
            st.rerun()

    personality = st.selectbox(
        "🤖 AI Personality",
        ["General AI", "Teacher", "Coding Assistant", "Interviewer"]
    )

    model = st.selectbox(
        "⚡ Gemini Model",
        ["gemini-flash-latest", "gemini-pro-latest"]
    )

    temperature = st.slider("🌡️ Creativity", 0.0, 2.0, 0.7, 0.1)

    st.divider()

    if st.session_state.messages:
        history = ""
        for msg in st.session_state.messages:
            history += f'{msg["role"]}: {msg["content"]}\n\n'
        st.download_button("📥 Download Chat", history, file_name="chat_history.txt")

# ----------------------------
# Title
# ----------------------------
st.title("🤖 EDITH")

# ----------------------------
# Display Messages
# ----------------------------
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# ----------------------------
# Chat Input
# ----------------------------
col_input, col_mic = st.columns([6, 1])

with col_mic:
    audio_value = st.audio_input("🎤", label_visibility="collapsed")

with col_input:
    prompt = st.chat_input("Ask me anything...")

if audio_value is not None:
    audio_bytes = audio_value.getvalue()
    if st.session_state.get("last_audio_bytes") != audio_bytes:
        st.session_state.last_audio_bytes = audio_bytes
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        try:
            transcribed = recognizer.recognize_google(audio_data)
            st.session_state.pending_prompt = transcribed
            st.rerun()
        except sr.UnknownValueError:
            st.error("Could not understand the audio. Try recording again.")
        except sr.RequestError as e:
            st.error(f"Speech recognition service error: {e}")

if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    personality_prompts = {
        "General AI": "You are EDITH. Be friendly. Answer clearly. Use simple English.",
        "Teacher": "You are an expert teacher. Explain every topic step by step. Use examples. Teach like a classroom instructor.",
        "Coding Assistant": "You are an expert Python programmer. Explain every line of code. Write clean Python code. Suggest best practices.",
        "Interviewer": "You are an interviewer. Ask one interview question at a time. Wait for the user's answer. Then provide feedback."
    }
    system_prompt = personality_prompts[personality]

    conversation = system_prompt + "\n\n"
    for msg in st.session_state.messages:
        conversation += f'{msg["role"]}: {msg["content"]}\n'

    with st.chat_message("assistant", avatar="🤖"):
        thinking = st.empty()
        thinking.info("🤔 Thinking...")
        placeholder = st.empty()

        try:
            response = client.models.generate_content_stream(
                model=model,
                contents=conversation
            )
            ai_response = ""
            for chunk in response:
                if chunk.text:
                    ai_response += chunk.text
                    placeholder.markdown(ai_response + "▌")
        except Exception as e:
            st.error(f"Gemini Error: {e}")
            st.stop()

    thinking.empty()
    placeholder.markdown(ai_response)

    try:
            tts = gTTS(text=ai_response, lang='en')
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            st.audio(audio_fp, format="audio/mp3")
    except Exception as e:
            st.caption(f"🔇 Voice output unavailable: {e}")

            st.session_state.messages.append({"role": "assistant", "content": ai_response})

    with open(f"chats/{st.session_state.chat_id}.json", "w") as file:
        json.dump(st.session_state.messages, file, indent=4)