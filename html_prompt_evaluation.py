# evaluate_prompts_k8.py
import time
import json
import csv
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document

# -----------------------------
# Paths / Output
# -----------------------------
INDEX_DIR = Path("chatbot_project") / "faiss_cs500_co50"
RESULTS_DIR = Path("html_prompt_eval_k8_results")
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "html_prompt_eval_k8.csv"
JSONL_PATH = RESULTS_DIR / "html_prompt_eval_k8_details.jsonl"

# -----------------------------
# Retrieval / Generation settings
# -----------------------------
K = 8
LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"
TEMPERATURE = 0.0
NUM_PREDICT = 512
NUM_CTX = 8192  # increase if your local model/tag supports more

# -----------------------------
# Questions
# -----------------------------
QUESTIONS: List[Dict[str, str]] = [
    {"id": "Q1", "question": "What is the credit value of CHEM2112?"},
    {"id": "Q2", "question": "How many second-year modules are available?"},
    {"id": "Q3", "question": "Are there any first-year modules assessed by 100% coursework?"},
    {"id": "Q4", "question": "Summarise what topics are covered across all first-year chemistry modules."},
    {"id": "Q5", "question": "Are there any modules specifically about nanotechnology?"},
    {"id": "Q6", "question": "How do you know which modules are second-year modules?"},
]

# -----------------------------
# Prompt Templates (exactly as requested)
# -----------------------------
STRUCTURED_SYSTEM = """You are a helpful assistant for answering university module catalog questions.

**Instructions:**
- Use the following context to answer the question.
- If the answer is not explicitly stated, reply: "I don't know."
- Keep the answer clear and concise.
- Where applicable, state the credit value or module name exactly as shown.
- Do not repeat the context text verbatim.
"""

STRUCTURED_USER = """**Question:** {question}

**Context:**
{context}

**Answer (one short paragraph or bullet points):**
"""

COUNT_SUMM_SYSTEM = """You are an expert assistant for answering detailed questions about university modules.

**Instructions:**
- Carefully review the context and identify any counts, credit values, or summaries requested.
- When asked to count or list items (e.g., modules), provide an exact number and a short list of names.
- If the answer cannot be determined, reply: "I don't know."
- Be precise and do not include unrelated information.
"""

COUNT_SUMM_USER = """**Question:** {question}

**Context:**
{context}

**Answer (include counts and short summaries if needed):**
"""

BULLETS_SYSTEM = """You are an assistant that answers questions about university module catalog entries.

**Instructions:**
- Use the provided context to answer.
- If you don't know, reply: "I don't know."
- Answer in 1–3 short bullet points.
- Do not include repeated context or unnecessary commentary.
"""

BULLETS_USER = """**Question:** {question}

**Context:**
{context}

**Answer in bullet points:**
"""

PROMPTS = {
    "structured": (STRUCTURED_SYSTEM, STRUCTURED_USER),
    "counting_summarisation": (COUNT_SUMM_SYSTEM, COUNT_SUMM_USER),
    "bullet_points": (BULLETS_SYSTEM, BULLETS_USER),
}

# -----------------------------
# Helpers
# -----------------------------
def load_faiss(index_path: Path):
    print(f"[INFO] Loading FAISS index from: {index_path}")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vs = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    print("[INFO] FAISS index loaded successfully.")
    # Quick diagnostics
    try:
        ntotal = vs.index.ntotal
        print(f"[INFO] FAISS vectors: {ntotal}")
    except Exception:
        pass
    try:
        doc_count = len(vs.docstore._dict)
        print(f"[INFO] Docstore documents: {doc_count}")
    except Exception:
        pass
    return vs

def build_context(docs: List[Document]) -> str:
    if not docs:
        return ""
    parts = []
    for d in docs:
        src = d.metadata.get("source_path", d.metadata.get("source", "unknown"))
        parts.append(f"[{src}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

def make_chain(system_tmpl: str, user_tmpl: str, llm: ChatOllama):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_tmpl),
        ("user", user_tmpl),
    ])
    return prompt | llm

def warm_up(vs, chain) -> None:
    print("[INFO] Warm-up: retrieval + generation...")
    try:
        _ = vs.as_retriever(search_kwargs={"k": 1}).get_relevant_documents("warm-up query")
    except Exception:
        pass
    try:
        _ = chain.invoke({"question": "Say 'Ready'.", "context": "This is a warm-up context."})
    except Exception:
        pass
    print("[INFO] Warm-up complete.")

# -----------------------------
# Main
# -----------------------------
def main():
    if not INDEX_DIR.exists():
        print(f"[ERROR] Index directory not found: {INDEX_DIR}")
        return

    print("[INFO] Initialising LLM...")
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        num_predict=NUM_PREDICT,
        num_ctx=NUM_CTX,
    )

    vs = load_faiss(INDEX_DIR)
    retriever = vs.as_retriever(search_kwargs={"k": K})

    # Build chains for each prompt template
    chains = {
        name: make_chain(sys_t, usr_t, llm)
        for name, (sys_t, usr_t) in PROMPTS.items()
    }

    # Warm-up (using one of the chains is enough to load the model)
    warm_up(vs, chains["structured"])

    print("[INFO] Starting evaluation...")
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f_csv, \
         open(JSONL_PATH, "w", encoding="utf-8") as f_jsonl:

        csv_writer = csv.writer(f_csv)
        csv_writer.writerow([
            "index_name", "k", "prompt_template",
            "question_id", "latency_total_sec",
            "latency_retrieval_sec", "latency_generation_sec",
            "retrieval_count"
        ])

        for tmpl_name, chain in chains.items():
            print(f"\n[INFO] Evaluating template: {tmpl_name} (k={K})")
            for i, q in enumerate(QUESTIONS, start=1):
                question = q["question"]
                print(f"  [INFO] {tmpl_name} | Q{i}/{len(QUESTIONS)}: {question}")

                # Retrieval timing
                t0 = time.perf_counter()
                docs = retriever.get_relevant_documents(question)
                t1 = time.perf_counter()
                retrieval_latency = t1 - t0

                context = build_context(docs)
                print(f"    [DEBUG] Retrieved {len(docs)} docs | Context length: {len(context)} chars")

                # Generation timing
                t2 = time.perf_counter()
                msg = {"question": question, "context": context}
                ans_msg = chain.invoke(msg)
                answer = getattr(ans_msg, "content", str(ans_msg)).strip()
                t3 = time.perf_counter()
                generation_latency = t3 - t2

                total_latency = t3 - t0

                # CSV summary
                csv_writer.writerow([
                    INDEX_DIR.name, K, tmpl_name,
                    q["id"], f"{total_latency:.3f}",
                    f"{retrieval_latency:.3f}",
                    f"{generation_latency:.3f}",
                    len(docs)
                ])

                # JSONL detail
                detail = {
                    "index_name": INDEX_DIR.name,
                    "k": K,
                    "prompt_template": tmpl_name,
                    "question_id": q["id"],
                    "question": question,
                    "answer": answer,
                    "latency_sec": {
                        "total": total_latency,
                        "retrieval": retrieval_latency,
                        "generation": generation_latency,
                    },
                    "retrieved_docs": [
                        {
                            "source": d.metadata.get("source_path", d.metadata.get("source", "unknown")),
                            "snippet": d.page_content[:700],
                            "metadata": {k2: v for k2, v in d.metadata.items() if k2 != "page_content"}
                        }
                        for d in docs
                    ],
                }
                f_jsonl.write(json.dumps(detail, ensure_ascii=False) + "\n")

    print(f"\n[OK] Wrote summary CSV -> {CSV_PATH}")
    print(f"[OK] Wrote detailed JSONL -> {JSONL_PATH}")
    print("[INFO] Prompt evaluation complete.")

if __name__ == "__main__":
    main()
