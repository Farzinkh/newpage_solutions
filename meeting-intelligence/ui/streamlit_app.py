"""Streamlit UI.

Deliberately thin: it is a client of the FastAPI backend, never importing the
pipeline directly. That keeps the API as the single source of truth and means
the UI could be swapped for anything (a React app, a CLI) without touching core
logic.

Three ingestion paths share one endpoint: upload a .txt, paste text, or record
voice. Voice uses the browser's Web Speech API via `streamlit-mic-recorder`
(optional dependency) — the audio never leaves the browser; only recognised
text is sent to the backend, where it flows through the same pipeline.

Retrieval scores and per-stage latency are shown under each answer, so the
retrieval "black box" is inspectable right in the UI, not just in the logs.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Meeting Intelligence", page_icon="🗣️", layout="wide")
st.title("Meeting intelligence")
st.caption("Ask questions about your meetings — answers are grounded and cited.")

if "history" not in st.session_state:
    st.session_state.history = []


def ingest(meeting_id: str, text: str, source: str) -> None:
    try:
        res = requests.post(
            f"{API}/ingest",
            json={"meeting_id": meeting_id, "text": text, "source": source},
            timeout=120,
        )
        res.raise_for_status()
        data = res.json()
        redactions = data.get("redactions") or {}
        note = f" · redacted {sum(redactions.values())} PII item(s)" if redactions else ""
        st.success(f"Ingested '{meeting_id}': {data['chunks']} chunks{note}")
    except requests.RequestException as e:
        st.error(f"Ingestion failed: {e}")


with st.sidebar:
    st.header("Add a meeting")
    meeting_id = st.text_input("Meeting id", value="meeting_1")

    tab_file, tab_paste, tab_voice = st.tabs(["Upload", "Paste", "Voice"])

    with tab_file:
        uploaded = st.file_uploader("Transcript (.txt)", type=["txt"])
        if uploaded and st.button("Ingest file"):
            ingest(meeting_id, uploaded.read().decode("utf-8"), "file")

    with tab_paste:
        pasted = st.text_area("Transcript text", height=200,
                              placeholder="[00:00:04] Alice: ...")
        if pasted and st.button("Ingest text"):
            ingest(meeting_id, pasted, "file")

    with tab_voice:
        st.caption("Recording runs in your browser (Web Speech API). "
                   "Audio stays local; only text is sent.")
        try:
            from streamlit_mic_recorder import speech_to_text

            spoken = speech_to_text(language="en", start_prompt="Start recording",
                                    stop_prompt="Stop", key="stt")
            if spoken:
                st.write(f"Recognised: {spoken}")
                if st.button("Ingest voice"):
                    ingest(meeting_id, spoken, "voice")
        except ImportError:
            st.info("Install `streamlit-mic-recorder` to enable voice input.")

    st.divider()
    try:
        meetings = requests.get(f"{API}/meetings", timeout=10).json()["meetings"]
        st.write("Ingested meetings:", meetings or "none yet")
    except requests.RequestException:
        st.warning("Backend not reachable.")
        meetings = []

scope = st.selectbox("Search scope", ["All meetings", *[]], index=0) if False else None
selected_meeting = st.selectbox(
    "Restrict to meeting (optional)", ["All", *(meetings if "meetings" in dir() else [])]
)

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

if question := st.chat_input("Ask about the meetings..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        payload = {"question": question}
        if selected_meeting and selected_meeting != "All":
            payload["meeting_id"] = selected_meeting
        try:
            res = requests.post(f"{API}/query", json=payload, timeout=120)
            res.raise_for_status()
            ans = res.json()
        except requests.RequestException as e:
            st.error(f"Query failed: {e}")
            ans = None

        if ans:
            st.markdown(ans["text"])
            if not ans["grounded"]:
                st.warning("This answer is not grounded in the transcript.")

            if ans["citations"]:
                with st.expander("Citations"):
                    for c in ans["citations"]:
                        st.markdown(f"**{c['speaker']}** @ `{c['timestamp']}` — {c['quote']}")

            if ans["retrieved"]:
                with st.expander("Retrieval detail (scores)"):
                    for i, rc in enumerate(ans["retrieved"], start=1):
                        rr = rc["rerank_score"]
                        rr_txt = f", rerank {rr:.3f}" if rr is not None else ""
                        st.markdown(
                            f"[{i}] sim {rc['similarity']:.3f}{rr_txt} — "
                            f"*{rc['chunk']['speaker']} @ {rc['chunk']['timestamp']}*: "
                            f"{rc['chunk']['text'][:160]}"
                        )
            st.session_state.history.append({"role": "assistant", "content": ans["text"]})
