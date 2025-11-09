# Finance House Policy Bot — Streamlit UI (CSS-free, theme-safe)
# Drop-in replacement focused on robust icons/avatars, visible citations, and query timing.

import os
import time
from typing import List, Dict, Any, Optional

import streamlit as st

# Optional: use Pillow so local PNGs can also serve as page_icon and chat avatars
try:
    from PIL import Image
except Exception:
    Image = None  # Fallback to emoji-only if Pillow isn't available

# --- RAG stack (kept minimal; works if deps and DB exist, otherwise degrades gracefully) ---
try:
    from langchain_core.documents import Document
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_cohere import CohereRerank
except Exception:
    Document = None
    ChatGoogleGenerativeAI = None
    Chroma = None
    SentenceTransformerEmbeddings = None
    CohereRerank = None

# --------------------------
# Configuration & constants
# --------------------------
APP_TITLE = "Finance House Policy Bot 🇦🇪"
DATA_DIR = "data"
DB_DIR = "db"
ASSETS_DIR = "assets"
AVATAR_PNG = os.path.join(ASSETS_DIR, "fh_avatar.png")  # 128×128 PNG recommended
LOGO_PNG = os.path.join(ASSETS_DIR, "fh_logo.png")      # wide transparent PNG
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RETRIEVER_K = 10
RERANK_TOP_N = 3

# --------------------------
# Helpers: assets & avatars
# --------------------------
def _load_png_or_emoji(path: str, emoji_fallback: str):
    """
    Returns a PIL.Image for a local PNG if present and Pillow is installed;
    otherwise returns the provided emoji string which Streamlit accepts as an icon/avatar.
    """
    if Image is not None and os.path.exists(path):
        try:
            return Image.open(path)
        except Exception:
            pass
    return emoji_fallback

def _assistant_avatar():
    return _load_png_or_emoji(AVATAR_PNG, "🤖")

def _page_icon():
    return _load_png_or_emoji(AVATAR_PNG, "🏦")

# --------------------------
# Streamlit page & sidebar
# --------------------------
st.set_page_config(
    page_title="Finance House Policy Bot",
    page_icon=_page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.caption(f"Streamlit {st.__version__}")  # quick visibility of Cloud runtime
    if os.path.exists(LOGO_PNG) and Image is not None:
        st.image(LOGO_PNG, use_container_width=True)
    elif os.path.exists(LOGO_PNG):
        st.image(LOGO_PNG, use_container_width=True)
    else:
        st.markdown("🏦 Finance House")
    st.markdown("Get accurate, citable answers from core company policies using a lightweight RAG pipeline.")

# --------------------------
# Lazy RAG components
# --------------------------
@st.cache_resource(show_spinner=False)
def _build_embeddings():
    if SentenceTransformerEmbeddings is None:
        return None
    try:
        return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def _load_vectorstore():
    if Chroma is None:
        return None
    if not os.path.isdir(DB_DIR):
        return None
    try:
        emb = _build_embeddings()
        if emb is None:
            return None
        return Chroma(persist_directory=DB_DIR, embedding_function=emb)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def _reranker():
    if CohereRerank is None:
        return None
    api_key = os.getenv("COHERE_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        return CohereRerank(api_key=api_key, model="rerank-english-v3.0")
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def _llm():
    if ChatGoogleGenerativeAI is None:
        return None
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    except Exception:
        return None

# --------------------------
# RAG answer function
# --------------------------
SYSTEM_PROMPT = (
    "You are an assistant that answers questions strictly using the provided company policy context. "
    "Quote exact lines where appropriate, and list short source identifiers in parentheses."
)

def _multi_queries(q: str) -> List[str]:
    """
    Simple multi-query generator; if an LLM is available, generate variants,
    else return basic heuristics.
    """
    llm = _llm()
    if llm is not None:
        try:
            prompt = (
                "Generate 3 short alternative phrasings of the following question to improve retrieval. "
                "Return each on a new line without numbering.\n\nQuestion: " + q
            )
            out = llm.invoke(prompt)
            text = getattr(out, "content", str(out))
            variants = [line.strip(" -•") for line in text.splitlines() if line.strip()]
            if variants:
                return [q] + variants[:3]
        except Exception:
            pass
    # Heuristic fallbacks
    return [q, f"What details are in: {q}?", f"Policy rules for: {q}", f"Definitions related to: {q}"]

def _retrieve(subqs: List[str]) -> List[Any]:
    """
    Retrieve documents for the sub-queries; returns a de-duplicated list of Documents.
    """
    vs = _load_vectorstore()
    if vs is None:
        return []
    retriever = vs.as_retriever(search_kwargs={"k": RETRIEVER_K})
    seen = set()
    out = []
    for sq in subqs:
        try:
            docs = retriever.get_relevant_documents(sq)
            for d in docs:
                key = getattr(d, "page_content", "")[:80] + str(getattr(d, "metadata", {}))
                if key not in seen:
                    seen.add(key)
                    out.append(d)
        except Exception:
            continue
    return out

def _apply_rerank(docs: List[Any], q: str) -> List[Any]:
    rr = _reranker()
    if rr is None or not docs:
        return docs[:RERANK_TOP_N]
    try:
        ranked = rr.rerank(query=q, documents=[getattr(d, "page_content", "") for d in docs], top_n=RERANK_TOP_N)
        # CohereRerank returns items with index; map back to docs
        chosen = []
        for item in ranked:
            idx = getattr(item, "index", None)
            if idx is not None and 0 <= idx < len(docs):
                chosen.append(docs[idx])
        return chosen or docs[:RERANK_TOP_N]
    except Exception:
        return docs[:RERANK_TOP_N]

def _synthesize_answer(q: str, context_docs: List[Any]) -> str:
    """
    Use LLM if available; otherwise compose a deterministic extractive answer.
    """
    llm = _llm()
    joined = "\n\n".join([getattr(d, "page_content", "") for d in context_docs])[:12000]
    if llm is not None and joined:
        try:
            prompt = (
                f"{SYSTEM_PROMPT}\n\nQuestion: {q}\n\nContext:\n{joined}\n\n"
                "Answer clearly and concisely, and include bracketed source tags like (POL-HR-004) where applicable."
            )
            out = llm.invoke(prompt)
            return getattr(out, "content", str(out))
        except Exception:
            pass
    if not context_docs:
        return "No policy context is available in the current knowledge base. Please upload or rebuild the vector store."
    # Simple extractive fallback
    snippet = getattr(context_docs[0], "page_content", "")
    return f"From the most relevant policy excerpt:\n\n> {snippet[:1000]}"

def _citations(docs: List[Any]) -> List[str]:
    cites = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        src = md.get("source") or md.get("file") or md.get("policy") or md.get("id") or "local-doc"
        cites.append(str(src))
    # De-duplicate, keep order
    seen = set()
    uniq = []
    for c in cites:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

def generate_answer(user_query: str) -> Dict[str, Any]:
    """
    Full pipeline: multi-query -> retrieve -> rerank -> synthesize -> meta
    Returns dict with keys: content (md string), meta (dict as used by renderer)
    """
    t0 = time.time()
    subqs = _multi_queries(user_query)
    raw_docs = _retrieve(subqs)
    top_docs = _apply_rerank(raw_docs, user_query)
    answer_md = _synthesize_answer(user_query, top_docs)
    elapsed_ms = int((time.time() - t0) * 1000)

    meta = {
        "elapsed_ms": elapsed_ms,
        "retrieved": len(raw_docs),
        "reranked": len(top_docs),
        "citations": _citations(top_docs),
        "intent": "policy Q&A",
        "subqueries": subqs,
        "retrieved_docs": [
            (getattr(d, "metadata", {}) or {}).get("title")
            or (getattr(d, "metadata", {}) or {}).get("source")
            or "doc"
            for d in raw_docs
        ],
        "reranked_docs": [
            (getattr(d, "metadata", {}) or {}).get("title")
            or (getattr(d, "metadata", {}) or {}).get("source")
            or "doc"
            for d in top_docs
        ],
    }
    return {"content": answer_md, "meta": meta}

# --------------------------
# Chat UI helpers
# --------------------------
def render_message(role: str, text: str, response_meta: Optional[Dict[str, Any]] = None):
    """
    Render a chat message with a theme-safe meta header and an expanded reasoning section.
    """
    avatar = _assistant_avatar() if role in ("assistant", "ai") else "👤"
    with st.chat_message(role, avatar=avatar):
        # Meta header row (always visible)
        if role in ("assistant", "ai") and response_meta:
            c1, c2, c3 = st.columns([1, 1, 3])
            with c1:
                if "elapsed_ms" in response_meta:
                    st.metric("Query ms", int(response_meta["elapsed_ms"]))
            with c2:
                if "retrieved" in response_meta or "reranked" in response_meta:
                    r = response_meta.get("retrieved", 0)
                    k = response_meta.get("reranked", 0)
                    st.caption(f"Docs: {k} of {r}")
            with c3:
                cites = response_meta.get("citations")
                if cites:
                    if isinstance(cites, (list, tuple)):
                        st.caption("Sources: " + "; ".join(str(x) for x in cites))
                    else:
                        st.caption(f"Source: {cites}")

        # Main content
        st.markdown(text)

        # Reasoning details
        if role in ("assistant", "ai") and response_meta:
            with st.expander("Show Reasoning 🧠", expanded=True):
                if "intent" in response_meta:
                    st.caption(f"Policy intent: {response_meta['intent']}")
                if response_meta.get("subqueries"):
                    st.markdown("#### Generated sub‑queries")
                    for q in response_meta["subqueries"]:
                        st.caption(f"• {q}")
                if response_meta.get("retrieved_docs"):
                    st.markdown("#### Retrieved documents (pre‑rank)")
                    for d in response_meta["retrieved_docs"]:
                        st.caption(f"• {d}")
                if response_meta.get("reranked_docs"):
                    st.markdown("#### Reranked top‑k")
                    for d in response_meta["reranked_docs"]:
                        st.caption(f"• {d}")

# --------------------------
# Page body
# --------------------------
st.title(APP_TITLE)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I'm the Finance House Policy Bot. How can I help you today?", "meta": None}
    ]

# Render history
for msg in st.session_state["messages"]:
    render_message(msg["role"], msg["content"], msg.get("meta"))

# Input
user_input = st.chat_input("Ask a question about a company policy…")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input, "meta": None})
    render_message("user", user_input, None)

    with st.spinner("Thinking…"):
        result = generate_answer(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": result["content"], "meta": result["meta"]})
    render_message("assistant", result["content"], result["meta"])
