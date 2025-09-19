# tabs_caching_with_dropdowns.py
from pathlib import Path
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

# ---------------- Configuration ----------------
st.set_page_config(page_title="Uni Module Chatbot (RAG)", page_icon="🎓")
st.title("🎓 University Module Chatbot (RAG)")

INDEX_DIR = Path("chatbot_project") / "faiss_cs500_co50"
FIXED_K = 8
MODEL_NAME = "llama3"
EMBED_MODEL = "nomic-embed-text"

BULLET_TEMPLATE = """You are an assistant that answers questions about university module catalog entries.

**Instructions:**
- Use the provided context to answer.
- If you don't know, reply: "I don't know."
- Answer in 1–3 short bullet points.
- Do not include repeated context or unnecessary commentary.

**Question:** {question}

**Context:**
{context}

**Answer in bullet points:**"""

# ---------------- Caching ----------------
@st.cache_resource(show_spinner=True)
def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)

@st.cache_resource(show_spinner=True)
def load_vectorstore(index_path: Path, _embeddings):
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at: {index_path.resolve()}")
    return FAISS.load_local(str(index_path), _embeddings, allow_dangerous_deserialization=True)

@st.cache_resource(show_spinner=True)
def make_retriever(_vs):
    return _vs.as_retriever(search_kwargs={"k": FIXED_K})

@st.cache_resource(show_spinner=True)
def build_chain():
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    prompt = ChatPromptTemplate.from_template(BULLET_TEMPLATE)
    return prompt | llm | StrOutputParser()

@st.cache_data(show_spinner=False)
def get_cached_sources(_docstore_dict):
    seen, sources = set(), []
    for d in _docstore_dict.values():
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        if src in seen: 
            continue
        seen.add(src)
        title = d.metadata.get("title")
        label = f"{title} — `{src}`" if title and title != src else f"`{src}`"
        sources.append(label)
    return sources

# ---------------- Sidebar ----------------
if "show_context" not in st.session_state:
    st.session_state.show_context = False
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True

with st.sidebar:
    st.header("ℹ Info")
    st.session_state.show_context = st.toggle("Show retrieved context", value=st.session_state.show_context)
    st.session_state.show_sources = st.toggle("Show document sources (if available)", value=st.session_state.show_sources)
    st.markdown("---")
    st.caption(
        f"Model: `{MODEL_NAME}`\n"
        f"Embedding: `{EMBED_MODEL}`\n"
        f"FAISS Index: `{INDEX_DIR}`\n"
        f"Top-K: `{FIXED_K}`"
    )

# ---------------- Load pipeline ----------------
try:
    embeddings = get_embeddings()
    vs = load_vectorstore(INDEX_DIR, embeddings)
    retriever = make_retriever(vs)
    chain = build_chain()
except Exception as e:
    st.error(f"❌ Error loading chatbot components:\n\n{e}")
    st.stop()

def format_sources(docs):
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        title = d.metadata.get("title")
        lines.append(f"{i}. {title} — `{src}`" if title and title != src else f"{i}. `{src}`")
    return "\n".join(lines)

# ---------------- Tabs ----------------
tab_chat, tab_sources, tab_about = st.tabs(["💬 Chat", "📁 Sources", "📖 About"])

with tab_chat:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask about a module (e.g., prerequisites for CHEM5600M)")
    if user_q:
        st.session_state.chat.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = retriever.get_relevant_documents(user_q)
                context_text = "\n\n".join(d.page_content for d in docs) if docs else ""
                response = chain.invoke({"question": user_q, "context": context_text}) if docs else "I don't know."
                st.markdown(response)

                if st.session_state.show_context and context_text:
                    with st.expander("📄 Retrieved context"):
                        st.markdown(context_text)

                if st.session_state.show_sources and docs:
                    with st.expander("📚 Sources"):
                        st.markdown(format_sources(docs))

        st.session_state.chat.append({"role": "assistant", "content": response})

with tab_sources:
    st.subheader("📁 Document Sources in Index")
    try:
        sources = get_cached_sources(vs.docstore._dict)
        st.markdown("\n".join([f"- {s}" for s in sources])) if sources else st.info("No source metadata found.")
    except Exception:
        st.info("No source metadata found.")

with tab_about:
    st.subheader("ℹ About This Chatbot")
    st.markdown("""
This assistant uses **Retrieval-Augmented Generation (RAG)** to answer questions about university Chemistry modules.

**Key Features:**
- Locally hosted with `llama3` and `FAISS`
- Uses `nomic-embed-text` for vector search
- Retrieves top-8 context chunks from saved module data
- Responds with concise bullet-point answers
""")
