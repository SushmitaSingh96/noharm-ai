# ui_app.py
import streamlit as st
from src.transcript import stt
from src.classification import label_conversation
from pathlib import Path

st.title("Helmit – Harm Classification Demo")

audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "flac"])
PROMPT_PATH = Path("src/prompt.txt")

if audio_file is not None:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        summary_prompt = f.read()
    st.audio(audio_file)
    temp_path = Path("temp_audio") / audio_file.name
    temp_path.parent.mkdir(exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())

    transcript = stt(str(temp_path))
    label, reason, raw = label_conversation(transcript, summary_prompt)
    st.json({"label": int(label), "reason": reason, "raw[DEBUG}]": raw})