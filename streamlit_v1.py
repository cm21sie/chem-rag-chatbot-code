# simple_deployment.py
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

# ---------------- Load core components ----------------
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
if not INDEX_DIR.exists():
    st.error(f"FAISS index not found at: {INDEX_DIR.resolve()}")
    st.stop()
vs = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
retriever = vs.as_retriever(search_kwargs={"k": FIXED_K})

llm = ChatOllama(model=MODEL_NAME, temperature=0)
prompt = ChatPromptTemplate.from_template(BULLET_TEMPLATE)
chain = prompt | llm | StrOutputParser()

# ---------------- Chat UI ----------------
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

    st.session_state.chat.append({"role": "assistant", "content": response})
