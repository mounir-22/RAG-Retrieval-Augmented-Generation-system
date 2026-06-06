"""
st_app.py — Streamlit UI
Foundation: github.com/rauni-iitr/RAG-Langchain-ChromaDB-OpenSourceLLM-Streamlit
Upgrades: Conversational memory · Section-aware source display · Dark theme
"""
import os
import pymupdf
import streamlit as st
from dotenv import load_dotenv
from rag import build_or_load_vectorstore, build_hybrid_retriever, inference

load_dotenv()

st.set_page_config(
    page_title="Document Intelligence",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #000000;
        --bg-soft: #111111;
        --line: #2a2a2a;
        --text: #ffffff;
        --muted: #b5b5b5;
    }
    .logo-wrap {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }
    .logo-mark {
        width: 38px;
        height: 38px;
        border: 2px solid var(--text);
        color: var(--text);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        letter-spacing: 0.5px;
        font-size: 14px;
    }
    .logo-text {
        color: var(--text);
        font-size: 22px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .logo-sub {
        color: var(--muted);
        margin-bottom: 14px;
    }
    .source-box {
        background-color: var(--bg-soft);
        border: 1px solid var(--line);
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 6px;
        font-size: 0.84em;
        color: var(--text);
        white-space: pre-wrap;
    }
    .source-meta {
        font-weight: 700;
        color: var(--muted);
        margin-bottom: 6px;
        font-size: 0.88em;
    }
    .badge {
        background: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333333;
        border-radius: 4px;
        padding: 2px 7px;
        font-size: 0.78em;
        margin-right: 4px;
    }
    .msg-block {
        margin: 8px 0;
        padding: 10px 14px;
        border-radius: 6px;
        border: 1px solid #2a2a2a;
        background: #111111;
        line-height: 1.6;
    }
    .msg-label {
        font-size: 0.72em;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #888888;
        margin-right: 8px;
        display: inline;
    }
    .msg-content {
        display: inline;
        color: #ffffff;
    }
    </style>
    <div class="logo-wrap">
        <div class="logo-mark">DI</div>
        <div class="logo-text">Document Intelligence</div>
    </div>
    <div class="logo-sub">Ask your documents and review source-backed answers</div>
    """,
    unsafe_allow_html=True,
)

st.title("Document Intelligence")
st.caption("Ask your documents anything. Every answer cites its source")


@st.cache_resource(show_spinner="Indexing documents and loading models…")
def init(language: str, selected_file: str):
    vectordb, chunks, stats = build_or_load_vectorstore(language=language, selected_files=[selected_file])
    retriever = build_hybrid_retriever(vectordb, chunks)
    return retriever, stats


@st.cache_data
def read_document_text(filename: str) -> str:
    path = os.path.join("documents", filename)
    if filename.endswith(".pdf"):
        pages = []
        pdf = pymupdf.open(path)
        for page_num, page in enumerate(pdf, start=1):
            pages.append(f"\n\n--- Page {page_num} ---\n\n{page.get_text('text')}")
        pdf.close()
        return "".join(pages)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@st.cache_data
def read_document_bytes(filename: str) -> bytes:
    path = os.path.join("documents", filename)
    with open(path, "rb") as f:
        return f.read()

all_docs = sorted([f for f in os.listdir("documents") if f.endswith((".txt", ".pdf"))])

language_choice = "fr"
selected_doc = None
show_full_doc = False

# Default mapping by filename hints; user can override manually
fr_candidates = [f for f in all_docs if "_fr" in f.lower() or " - fr" in f.lower() or "french" in f.lower()]
en_candidates = [f for f in all_docs if "_en" in f.lower() or " - en" in f.lower() or "english" in f.lower()]

default_doc = all_docs[0] if all_docs else None
default_fr = fr_candidates[0] if fr_candidates else default_doc
default_en = en_candidates[0] if en_candidates else default_doc


with st.sidebar:
    st.link_button(
        "View Public GitHub Repository",
        "https://github.com/mounir-22/RAG-Retrieval-Augmented-Generation-system",
        use_container_width=True,
    )
    st.divider()
    st.markdown("**Language / Langue**")
    language_choice = st.radio(
        "Choose language",
        options=["fr", "en"],
        format_func=lambda x: "Francais" if x == "fr" else "English",
        horizontal=True,
    )

    if not all_docs:
        st.error("No .txt or .pdf files found in documents folder.")

    language_doc_default = default_fr if language_choice == "fr" else default_en
    if all_docs:
        default_index = all_docs.index(language_doc_default) if language_doc_default in all_docs else 0
        selected_doc = st.selectbox(
            "Document for selected language",
            all_docs,
            index=default_index,
            key=f"selected_doc_{language_choice}",
        )

    st.header("Indexed Documents")
    for fname in all_docs:
        st.markdown(f"**{fname}**")

    if selected_doc:
        st.caption(f"Active: **{selected_doc}**")

    if all_docs:
        st.divider()
        st.markdown("**Document Viewer**")
        show_full_doc = st.toggle("Show full file", value=False, key="show_full_file")

        if selected_doc and selected_doc.endswith(".pdf"):
            st.download_button(
                label="Download selected PDF",
                data=read_document_bytes(selected_doc),
                file_name=selected_doc,
                mime="application/pdf",
                use_container_width=True,
            )

    st.divider()
    st.markdown("**Download all PDFs**")
    for doc in all_docs:
        if doc.endswith(".pdf"):
            st.download_button(
                label=f"Download {doc}",
                data=read_document_bytes(doc),
                file_name=doc,
                mime="application/pdf",
                key=f"download_{doc}",
                use_container_width=True,
            )

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("- Hybrid Retrieval (Vector + BM25)")
    st.markdown("- Cross-Encoder Reranking")
    st.markdown("- Conversational Memory")
    st.markdown("- GPT-3.5-turbo")
    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if selected_doc:
    retriever, file_stats = init(language_choice, selected_doc)
else:
    retriever, file_stats = None, {"files": [], "chunks": 0}

if "messages" not in st.session_state:
    st.session_state.messages = []


if show_full_doc and selected_doc:
    with st.expander(f"Full Document: {selected_doc}", expanded=True):
        st.text(read_document_text(selected_doc))


def render_sources(source_docs):
    if not source_docs:
        return
    with st.expander("Sources used"):
        for i, doc in enumerate(source_docs, 1):
            fname = doc.metadata.get("source", "unknown")
            section = doc.metadata.get("section", "General")
            chunk_idx = doc.metadata.get("chunk_index", "?")
            page_num = doc.metadata.get("page")
            page_badge = f'<span class="badge">page {page_num}</span>' if page_num else ""
            st.markdown(
                f'<div class="source-box">'
                f'<div class="source-meta">'
                f'<span class="badge">#{i}</span>'
                f'<span class="badge">Source {fname}</span>'
                f'<span class="badge">§ {section}</span>'
                f'{page_badge}'
                f'<span class="badge">chunk {chunk_idx}</span>'
                f'</div>'
                f'{doc.page_content}'
                f'</div>',
                unsafe_allow_html=True,
            )


for msg in st.session_state.messages:
    label = "you:" if msg["role"] == "user" else "assistant:"
    st.markdown(
        f'<div class="msg-block"><span class="msg-label">{label}</span>'
        f'<span class="msg-content">{msg["content"]}</span></div>',
        unsafe_allow_html=True,
    )
    if msg["role"] == "assistant" and msg.get("sources"):
        render_sources(msg["sources"])

user_input = st.chat_input("Ask anything about your documents…")

if user_input:
    if retriever is None:
        st.error("No active document selected.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(
        f'<div class="msg-block"><span class="msg-label">you:</span>'
        f'<span class="msg-content">{user_input}</span></div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Searching documents…"):
        stream, source_docs = inference(
            user_input,
            retriever,
            chat_history=st.session_state.messages[:-1],
            language=language_choice,
        )
    response = st.write_stream(stream)
    st.markdown(
        f'<div class="msg-block"><span class="msg-label">assistant:</span>'
        f'<span class="msg-content">{response}</span></div>',
        unsafe_allow_html=True,
    )
    render_sources(source_docs)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": source_docs}
    )
