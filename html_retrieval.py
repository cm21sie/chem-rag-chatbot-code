# retrieval.py

import re
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------------
# CONFIG
# --------------------------
MODEL_NAME = "nomic-embed-text"
FAISS_INDEX_FILE = "html_faiss_index.index"
QUERY = "What is the credit value of CHEM2112?"
TOP_K = 12  # number of documents to retrieve (use a bit higher for robustness)

# --------------------------
# HELPERS
# --------------------------
def norm_code(s: str) -> str:
    """Uppercase and remove spaces: 'chem 2112' -> 'CHEM2112'."""
    return re.sub(r"\s+", "", s.upper())

def extract_query_code(q: str):
    """Return 'CHEM1234' if present in query (handles spaces), else None."""
    m = re.search(r"\bCHEM\s*([0-9]{4})\b", q, flags=re.I)
    return f"CHEM{m.group(1)}" if m else None

# --------------------------
# LOAD EMBEDDING MODEL
# --------------------------
embedding_model = OllamaEmbeddings(model=MODEL_NAME)

# --------------------------
# LOAD FAISS INDEX
# --------------------------
if not Path(FAISS_INDEX_FILE).exists():
    raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_FILE}")

vectorstore = FAISS.load_local(
    FAISS_INDEX_FILE,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# --------------------------
# RETRIEVE (robust for module codes)
# --------------------------
raw_query = QUERY
target_code = extract_query_code(raw_query)          # e.g., 'CHEM2112' or None
norm_q_code = norm_code(target_code) if target_code else None
boost_keywords = []
if target_code:
    boost_keywords = [norm_q_code, f"{norm_q_code[:4]} {norm_q_code[4:]}", "credits"]

# MMR tends to diversify while keeping relevance
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": TOP_K, "fetch_k": max(40, TOP_K * 4), "lambda_mult": 0.5}
)

# Dense retrieval with keyword help if we have a code
dense_query = raw_query if not boost_keywords else f"{raw_query}\nKeywords: {' '.join(boost_keywords)}"
docs_dense = retriever.invoke(dense_query)

# Exact-match docstore scan (metadata + content)
exact_hits = []
if target_code:
    code_4, code_num = target_code[:4], target_code[4:]
    content_pat = re.compile(rf"\b{code_4}\s*{code_num}\b", flags=re.I)

    # Note: accessing _dict is fine for debugging/robustness here
    for _id, d in vectorstore.docstore._dict.items():
        meta_code = norm_code(d.metadata.get("module_code", ""))
        if meta_code == norm_q_code or content_pat.search(d.page_content or ""):
            exact_hits.append(d)

# Merge exact hits first, then dense (dedup by (doc_id, chunk_id))
seen = set()
docs = []
for d in exact_hits + docs_dense:
    key = (d.metadata.get("doc_id"), d.metadata.get("chunk_id"))
    if key not in seen:
        docs.append(d)
        seen.add(key)

# If a target code is present, keep only that module in context (if any found)
if target_code:
    filtered = [d for d in docs if norm_code(d.metadata.get("module_code", "")) == norm_q_code]
    if filtered:
        docs = filtered

# --------------------------
# DIRECT CREDIT-LINE EXTRACTION (no LLM if we can avoid it)
# --------------------------
credit_line = None
credit_pat = re.compile(r"\bcredit[s]?:?\s*(\d{1,3})\b", flags=re.I)
code_line_pat = None
if target_code:
    code_line_pat = re.compile(rf"\bCHEM\s*{target_code[4:]}\b", flags=re.I)

for doc in docs:
    for line in doc.page_content.splitlines():
        # If asking for a specific code, prefer lines mentioning it
        if code_line_pat and not code_line_pat.search(line):
            continue
        m = credit_pat.search(line)
        if m:
            credit_line = f"{target_code} credits: {m.group(1)}"
            break
    if credit_line:
        break

# --------------------------
# IF FOUND, PRINT DIRECTLY
# --------------------------
if credit_line:
    print(f"Answer: {credit_line}")
else:
    # --------------------------
    # FALLBACK TO LLaMA3 VIA CHAT PROMPT (invoke API, no deprecation warnings)
    # --------------------------
    system_prompt = SystemMessagePromptTemplate.from_template(
        """You are a helpful and precise assistant for answering questions about university module catalogue entries.
Instructions:
- Use the provided context to answer the question.
- If the answer is not explicitly stated, reply with: "I don't know."
"""
    )
    human_prompt = HumanMessagePromptTemplate.from_template(
        "Context information is provided below. Use it to answer the question concisely.\n\n{context}\n\nQuestion: {question}"
    )
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    # Keep context tight; first few merged docs should include exact module if present
    context_text = "\n\n---\n\n".join([doc.page_content.strip() for doc in docs[:8]])

    # Load LLaMA3 model
    llm = OllamaLLM(model="llama3", temperature=0)

    # Build chain using new API
    chain = prompt | llm | StrOutputParser()

    # Run LLM via invoke()
    answer = chain.invoke({"context": context_text, "question": raw_query})
    print(f"Answer: {answer}")
