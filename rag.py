"""
rag.py — Core RAG logic
Foundation: github.com/rauni-iitr/RAG-Langchain-ChromaDB-OpenSourceLLM-Streamlit
Upgrades: Hybrid Search (Vector + BM25) · Cross-Encoder Reranking · Section
          Metadata · Conversational Memory · Privacy-Analyst Prompt
LLM: gpt-3.5-turbo via OpenAI API
"""
import os
import re
import hashlib
import pymupdf
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DOCUMENTS_DIR = "./documents"
CHROMA_DIR = "./chroma_db"
COLLECTION_BASE_NAME = "doc_intelligence_bilingual_v4"

# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE_FR = """Tu es un analyste expert en protection des donnees.

Reponds en francais, de facon claire et structuree, en t'appuyant strictement
sur les extraits fournis.

Consignes:
1. Commence par une reponse directe (1-2 phrases).
2. Ajoute ensuite une explication detaillee (2-5 phrases) avec les nuances,
conditions, exceptions et base juridique mentionnees.
3. Si la question porte sur des notions juridiques (consentement, base legale,
transfert, finalite, droits), explique comment le texte les formule.
4. Termine par une phrase de synthese pratique.
5. Utilise UNIQUEMENT le contexte fourni. N'invente rien.
6. Si le contexte ne permet pas de repondre clairement, dis-le explicitement.
{chat_history}
Extraits du document:
{context}

Question utilisateur: {question}

Reponse detaillee:"""

PROMPT_TEMPLATE_EN = """You are an expert data protection policy analyst.

Answer in English, clearly and in a structured way, based strictly on provided excerpts.

Instructions:
1. Start with a direct answer (1-2 sentences).
2. Then provide a detailed explanation (2-5 sentences) with nuances,
conditions, exceptions, and legal basis mentioned in the text.
3. For legal concepts (consent, legal basis, transfer, purpose, rights), explain
how the document formulates them.
4. End with one practical takeaway sentence.
5. Use ONLY the provided context. Do not invent facts.
6. If context is insufficient, explicitly say so.
{chat_history}
Document excerpts:
{context}

User question: {question}

Detailed answer:"""

RESEARCH_PROMPT_TEMPLATE_FR = """Tu es un analyste expert en protection des donnees.

L'utilisateur demande un resultat exhaustif (tous, toutes, lister, enumerer,
identifier toutes les occurrences).

Consignes:
1. Reponds en francais.
2. Identifie toutes les occurrences pertinentes dans les extraits fournis.
3. Ne fais pas un resume general: retourne une liste numerotee structuree.
4. Pour chaque element, utilise ce format:
   - Section: <nom de section>
   - Ce que dit le texte: <1-2 phrases>
5. Si plusieurs extraits concernent la meme section, fusionne-les.
6. Si la couverture est partielle, indique explicitement:
   "Cette liste est basee sur les extraits recuperes et peut ne pas inclure des
   sections non recuperees."
7. Utilise UNIQUEMENT le contexte fourni.
{chat_history}
Extraits du document:
{context}

Question utilisateur: {question}

Liste structuree:"""

RESEARCH_PROMPT_TEMPLATE_EN = """You are an expert data protection policy analyst.

The user asks for an exhaustive result (all/every/list/enumerate/find all).

Instructions:
1. Answer in English.
2. Identify every relevant occurrence in provided excerpts.
3. Do NOT provide a generic summary; return a structured numbered list.
4. For each item, use:
    - Section: <section name>
    - What the text says: <1-2 sentence explanation>
5. Merge items that belong to the same section.
6. If coverage may be partial, explicitly state:
    "This list is based on retrieved excerpts and may miss sections not retrieved."
7. Use ONLY provided context.
{chat_history}
Document excerpts:
{context}

User question: {question}

Structured list:"""

# ── Module-level reranker singleton (loaded once per process) ─────────────────
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("BAAI/bge-reranker-base")
        except Exception:
            _reranker = False  # mark as unavailable, skip next time
    return _reranker if _reranker else None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _detect_section(text: str) -> str:
    """Heuristic: first short, non-sentence line in the chunk = section heading."""
    for line in text.strip().split("\n")[:6]:
        line = line.strip()
        if line and len(line) < 80 and not line.endswith(".") and len(line.split()) <= 12:
            return line
    return "General"


def _load_documents_for_file(fname: str) -> list:
    """Load a single .txt or native text-based .pdf file into Document objects."""
    path = os.path.join(DOCUMENTS_DIR, fname)
    if fname.endswith(".txt"):
        loader = TextLoader(path, encoding="utf-8")
        return loader.load()
    if fname.endswith(".pdf"):
        # Native PDF text extraction only (no OCR, no scanned-doc processing)
        docs = []
        pdf = pymupdf.open(path)
        for page_num, page in enumerate(pdf, start=1):
            page_text = page.get_text("text")
            if page_text and page_text.strip():
                docs.append(
                    Document(
                        page_content=page_text,
                        metadata={"source": fname, "page": page_num},
                    )
                )
        pdf.close()
        return docs
    return []


def _load_and_chunk_documents(allowed_files=None) -> list:
    """
    Load all .txt and native text-based .pdf files and split into chunks.
    Filters out heading-only / near-empty chunks (< 80 chars of real content).
    Attaches metadata: source filename + detected section heading + page (for PDF).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks = []
    allowed = set(allowed_files) if allowed_files else None
    for fname in sorted(os.listdir(DOCUMENTS_DIR)):
        if not fname.endswith((".txt", ".pdf")):
            continue
        if allowed is not None and fname not in allowed:
            continue

        docs = _load_documents_for_file(fname)
        if not docs:
            continue

        chunks = splitter.split_documents(docs)
        idx = 0
        for chunk in chunks:
            # Skip heading-only or near-empty chunks
            text = chunk.page_content.strip()
            if len(text) < 80 or text.count(" ") < 8:
                continue
            chunk.metadata["source"] = fname
            chunk.metadata["section"] = _detect_section(text)
            chunk.metadata["chunk_index"] = idx
            all_chunks.append(chunk)
            idx += 1
    return all_chunks


def _build_collection_name(language: str, selected_files: list) -> str:
    files_key = "|".join(sorted(selected_files))
    digest = hashlib.md5(files_key.encode("utf-8")).hexdigest()[:10]
    return f"{COLLECTION_BASE_NAME}_{language}_{digest}"


def _is_research_query(query: str) -> bool:
    """Detect exhaustive/list-style queries that need high-recall retrieval."""
    triggers = [
        r"\ball\b",
        r"\bevery\b",
        r"\blist\b",
        r"\benumerate\b",
        r"\bfind all\b",
        r"\ball places\b",
        r"\ball sections\b",
        r"\bevery mention\b",
        r"\bidentify sections\b",
        r"\btous\b",
        r"\btoutes\b",
        r"\blister\b",
        r"\benumerer\b",
        r"\bidentifier\b",
        r"\btoutes les sections\b",
        r"\btoutes les occurrences\b",
    ]
    q = query.lower()
    return any(re.search(pattern, q) for pattern in triggers)


def _set_retriever_k(retriever, k: int) -> None:
    """Update k for both BM25 and vector retrievers inside EnsembleRetriever."""
    for sub_retriever in getattr(retriever, "retrievers", []):
        if hasattr(sub_retriever, "k"):
            sub_retriever.k = k
        if hasattr(sub_retriever, "search_kwargs") and isinstance(sub_retriever.search_kwargs, dict):
            sub_retriever.search_kwargs["k"] = k


def _dedupe_docs(docs: list, max_docs: int) -> list:
    """Deduplicate by source/section/content so research mode keeps broad coverage."""
    seen = set()
    unique_docs = []
    for doc in docs:
        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("section", ""),
            doc.page_content.strip()[:240],
        )
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)
        if len(unique_docs) >= max_docs:
            break
    return unique_docs


# ── Public API ────────────────────────────────────────────────────────────────
def build_or_load_vectorstore(language: str = "fr", selected_files=None) -> tuple:
    """
    Build ChromaDB from selected .txt/.pdf files in /documents, or load existing index.
    Always re-chunks from disk (fast, needed for BM25 which cannot be persisted).
    Returns (vectordb, chunks, file_stats).
    """
    embeddings = _get_embeddings()
    available_files = sorted(f for f in os.listdir(DOCUMENTS_DIR) if f.endswith((".txt", ".pdf")))
    selected = [f for f in (selected_files or available_files) if f in available_files]
    chunks = _load_and_chunk_documents(allowed_files=selected)

    if not chunks:
        raise FileNotFoundError(f"No supported documents found in {DOCUMENTS_DIR}")

    collection_name = _build_collection_name(language, selected)

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        chunk_count = vectordb._collection.count()
        # Fresh index for this collection name means 0 items → rebuild
        if chunk_count == 0:
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_DIR,
                collection_name=collection_name,
            )
            chunk_count = len(chunks)
    else:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
            collection_name=collection_name,
        )
        chunk_count = len(chunks)

    return vectordb, chunks, {"files": selected, "chunks": chunk_count, "language": language}


def build_hybrid_retriever(vectordb, chunks) -> EnsembleRetriever:
    """
    Hybrid Retrieval: 60% vector similarity + 40% BM25 keyword search.
    Fetches top-20 candidates for reranking.
    """
    # Rationale: policy/legal questions benefit from two retrieval signals.
    # - BM25 captures exact legal terms (e.g., "lawful basis", "controller").
    # - Vector similarity captures paraphrases and semantic intent.
    # A weighted ensemble gives better first-stage recall than either alone.
    vector_ret = vectordb.as_retriever(search_kwargs={"k": 20})
    bm25_ret = BM25Retriever.from_documents(chunks)
    bm25_ret.k = 20
    return EnsembleRetriever(
        retrievers=[bm25_ret, vector_ret],
        weights=[0.4, 0.6],
    )


def _rerank(query: str, docs: list, top_k: int = 4) -> list:
    """
    Cross-Encoder reranking: score every (query, chunk) pair, keep top_k.
    Falls back to first top_k results if reranker is unavailable.
    """
    reranker = _get_reranker()
    if reranker is None:
        # Fail-open behavior is intentional for production resilience:
        # retrieval still returns grounded context even if model download/
        # initialization fails in constrained environments.
        return docs[:top_k]
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def inference(query: str, retriever, chat_history: list = None, language: str = "fr") -> tuple:
    """
    Full pipeline with query-aware search mode:
    - Normal Q&A: top-20 retrieve -> rerank -> top-4
    - Research Mode: top-30 retrieve -> dedupe -> top-14
    Returns (stream_generator, source_docs).
    """
    research_mode = _is_research_query(query)

    # 1) Retrieval depth adapts to user intent:
    #    normal Q&A keeps precision high, research mode boosts recall.
    retrieval_k = 30 if research_mode else 20
    _set_retriever_k(retriever, retrieval_k)
    candidates = retriever.invoke(query)

    # 2. Mode-specific selection strategy
    if research_mode:
        # Favor recall for exhaustive questions
        source_docs = _dedupe_docs(candidates, max_docs=14)
    else:
        source_docs = _rerank(query, candidates, top_k=4)

    # 3. Build context with section labels
    context = "\n\n---\n\n".join(
        f"[{doc.metadata.get('section', 'General')}]\n{doc.page_content}"
        for doc in source_docs
    )

    # 4. Format conversational memory (last 3 turns)
    history_text = ""
    if chat_history:
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        history_text = f"\nPrevious conversation:\n{history_text}"

    # 5. Stream LLM response
    llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True, temperature=0)
    if language == "en":
        active_prompt = RESEARCH_PROMPT_TEMPLATE_EN if research_mode else PROMPT_TEMPLATE_EN
    else:
        active_prompt = RESEARCH_PROMPT_TEMPLATE_FR if research_mode else PROMPT_TEMPLATE_FR
    prompt = PromptTemplate(
        template=active_prompt,
        input_variables=["context", "question", "chat_history"],
    )
    chain = prompt | llm | StrOutputParser()

    stream = chain.stream({"context": context, "question": query, "chat_history": history_text})
    return stream, source_docs
