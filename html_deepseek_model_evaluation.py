# deepseek_model_evaluation.py
import time
import csv
import json
import re
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Files ---
CSV_OUTPUT_FILE = "html_deepseek_model_results.csv"
JSON_OUTPUT_FILE = "html_deepseek_model_output_examples.json"

# --- Model / Index settings ---
EMBEDDINGS_MODEL = "nomic-embed-text"      # must match the model used to build the FAISS index
FAISS_INDEX_PATH = "html_faiss_index.index"  # new index path
LLM_MODEL = "deepseek-r1:latest"
TOP_K = 6
FETCH_K = 40

# --- Prompt (updated for best-from-context answers + evidence) ---
system_prompt = SystemMessagePromptTemplate.from_template(
    """You are a careful assistant answering questions about university modules using ONLY the provided context.

GENERAL PRINCIPLES
- If the answer is not explicitly supported by the context, respond exactly with: "I don't know."
- Prefer information from module HTML catalogue pages over text summaries if both are present.
- When multiple values appear, choose the one most specific to the asked module; if conflicting, state the range or say "I don't know."
- Extract numeric values (credits, percentages, hours) carefully; do not infer or round.
- Be concise and precise. Do not include hidden reasoning steps.
- If aggregating across modules, summarise patterns accurately without speculation.

OUTPUT FORMAT
Return:
1) A one- or two-sentence **Answer** directly addressing the question.
2) A short **Evidence** section with up to 2 quoted lines from the context that support the answer. Include source_file and module_code if available.

Example style:
Answer: 10 credits.
Evidence:
- "Credits: 10" (source: CHEM2112.html | CHEM2112)
- "Total hours (100hr per 10 credits) 200" (source: CHEM2112.html | CHEM2112)
"""
)

human_prompt_template = HumanMessagePromptTemplate.from_template(
    """Context:
{context}

Question: {question}

Instructions:
- Use only the context above.
- If the question mentions a specific module (e.g., CHEM1234), prefer lines from that module.
- Quote short, verbatim evidence lines if possible (≤ 20 words each).
- If the context is insufficient, reply exactly: "I don't know."

Return your response in this format:

Answer: <your best supported answer>

Evidence:
- "<short quote 1>" (source: <source_file> | <module_code>)
- "<short quote 2>" (source: <source_file> | <module_code>)"""
)

prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt_template])

# --- Embeddings & FAISS ---
print("Loading embeddings and FAISS index...")
embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Retriever (use invoke API; MMR for more robust coverage)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": 0.5}
)
print("FAISS index loaded.\n")

# --- LLM ---
llm = OllamaLLM(model=LLM_MODEL, temperature=0)

# --- RAG chain ---
rag_chain = prompt | llm | StrOutputParser()

# --- Questions ---
QUESTIONS = [
    "What is the credit value of CHEM2112?",
    "How many second-year modules are available?",
    "Are there any first-year modules assessed by 100% coursework?",
    "Summarise what topics are covered across all first-year chemistry modules.",
    "Are there any modules specifically about nanotechnology?",
    "How do you know which modules are second-year modules?"
]

# --- Helpers: module-code aware boost (optional but recommended) ---
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.upper())

def _extract_code(q: str):
    m = re.search(r"\bCHEM\s*([0-9]{4})\b", q, flags=re.I)
    return f"CHEM{m.group(1)}" if m else None

def _exact_hits_for_code(vstore, code: str):
    """Find exact matches in FAISS docstore by metadata or content (handles CHEM 2112 spacing)."""
    code_norm = _norm(code)
    code4, num = code_norm[:4], code_norm[4:]
    pat = re.compile(rf"\b{code4}\s*{num}\b", flags=re.I)
    hits = []
    for _id, d in vstore.docstore._dict.items():
        meta_code = _norm(d.metadata.get("module_code", ""))
        if meta_code == code_norm or pat.search(d.page_content or ""):
            hits.append(d)
    return hits

def _prefer_html(docs):
    return sorted(
        docs,
        key=lambda d: (
            0 if d.metadata.get("source_file","").lower().endswith(".html") else 1,
            d.metadata.get("source_file","")
        )
    )

def _merge_exact_then_dense(exact, dense):
    seen = set()
    merged = []
    for d in _prefer_html(exact) + dense:
        key = (d.metadata.get("doc_id"), d.metadata.get("chunk_id"))
        if key not in seen:
            merged.append(d)
            seen.add(key)
    return merged

# --- Function to answer a question ---
def answer_question(question, idx=None, top_n_docs=3, max_chars_per_doc=1000):
    if idx is not None:
        print(f"\n[{idx+1}/{len(QUESTIONS)}] Processing question: {question}")
    else:
        print(f"\nProcessing question: {question}")

    start_time = time.time()

    # Retrieve docs (invoke, not deprecated)
    print("Retrieving relevant documents...")
    retrieved_docs = retriever.invoke(question)
    print(f"Retrieved {len(retrieved_docs)} documents")

    # If a module code is in the query, prepend exact matches to ensure the right module appears
    code = _extract_code(question)
    if code:
        exact = _exact_hits_for_code(vectorstore, code)
        if exact:
            retrieved_docs = _merge_exact_then_dense(exact, retrieved_docs)
            # Optional: keep only that module if present
            only_code = [d for d in retrieved_docs if _norm(d.metadata.get("module_code","")) == _norm(code)]
            if only_code:
                retrieved_docs = only_code + [d for d in retrieved_docs if d not in only_code]

    # Keep top N docs and truncate for context size
    top_docs = retrieved_docs[:top_n_docs]
    context_text = "\n---\n".join([doc.page_content[:max_chars_per_doc] for doc in top_docs])
    print(f"Context length: {len(context_text)} characters")

    # Ask the model
    try:
        print("Sending context to model...")
        answer = rag_chain.invoke({"context": context_text, "question": question})
    except Exception as e:
        print(f"Error while querying model: {e}")
        answer = "Error"

    latency = time.time() - start_time
    print(f"Answer received in {latency:.2f} seconds: {answer[:120]}{'...' if len(answer)>120 else ''}")

    return retrieved_docs, context_text, answer, latency

# --- Run evaluation ---
results = []
example_outputs = []

for i, q in enumerate(QUESTIONS):
    retrieved_docs, context, answer, latency = answer_question(q, i)
    results.append({
        "question": q,
        "context": context,
        "answer": answer,
        "latency": round(latency, 3)
    })

    # Save example outputs with doc ids/chunks for traceability
    example_outputs.append({
        "question": q,
        "answer": answer,
        "retrieved_docs": [
            {
                "source_file": d.metadata.get("source_file", ""),
                "module_code": d.metadata.get("module_code", ""),
                "doc_id": d.metadata.get("doc_id", ""),
                "chunk_id": d.metadata.get("chunk_id", "")
            }
            for d in retrieved_docs[:3]
        ]
    })

# --- Save CSV ---
print("\nSaving CSV results...")
with open(CSV_OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "context", "answer", "latency"])
    writer.writeheader()
    writer.writerows(results)
print(f"CSV saved to {CSV_OUTPUT_FILE}")

# --- Save JSON ---
print("Saving JSON example outputs...")
with open(JSON_OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(example_outputs, f, indent=2)
print(f"JSON saved to {JSON_OUTPUT_FILE}\n")
print("Evaluation complete!")
