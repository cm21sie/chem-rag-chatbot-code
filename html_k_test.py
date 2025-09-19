import time
import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document

# -----------------------------
# File Locations
# -----------------------------
INDEX_DIR = Path("chatbot_project") / "faiss_cs500_co50"  # full path to index folder
RESULTS_DIR = Path("html_k_evaluation_results")
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "faiss_cs500_co50_k_eval.csv"
JSONL_PATH = RESULTS_DIR / "faiss_cs500_co50_k_eval_details.jsonl"

# -----------------------------
# Retrieval / Generation settings
# -----------------------------
K_VALUES = [1, 3, 5, 8, 10]
LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"
TEMPERATURE = 0.0
NUM_PREDICT = 512

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

SYSTEM_PROMPT = """You are a helpful, precise university information assistant.
Answer ONLY from the provided context. If the answer is not in the context, say you don't know.
Be concise and factual."""

USER_TEMPLATE = """Question:
{question}

Use ONLY this context (snippets from course/module pages). If insufficient, say you don't know:

{context}"""

def parse_cs_co(dirname: str):
    m = re.match(r"faiss_cs(\d+)_co(\d+)", dirname)
    cs = int(m.group(1)) if m else None
    co = int(m.group(2)) if m else None
    return cs, co

def load_faiss(index_path: Path):
    print(f"[INFO] Loading FAISS index from: {index_path}")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vs = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    print("[INFO] FAISS index loaded successfully.")
    return vs

def build_context(docs: List[Document]) -> str:
    if not docs:
        return ""
    parts = []
    for d in docs:
        src = d.metadata.get("source_path", d.metadata.get("source", "unknown"))
        parts.append(f"[{src}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

def warm_up(vs, chain) -> None:
    print("[INFO] Running warm-up to avoid cold-start latency...")
    try:
        _ = vs.as_retriever(search_kwargs={"k": 1}).get_relevant_documents("warm-up query")
    except Exception:
        pass
    try:
        _ = chain.invoke({"question": "Say 'Ready'.", "context": "This is a warm-up context."})
    except Exception:
        pass
    print("[INFO] Warm-up complete.")

def main():
    print("[INFO] Initialising LLM and prompt chain...")
    llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE, num_predict=NUM_PREDICT)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("user", USER_TEMPLATE)])
    chain = prompt | llm

    if not INDEX_DIR.exists():
        print(f"[ERROR] Specified index directory not found: {INDEX_DIR}")
        return

    cs, co = parse_cs_co(INDEX_DIR.name)
    vs = load_faiss(INDEX_DIR)

    warm_up(vs, chain)

    print("[INFO] Starting evaluation...")
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f_csv, \
         open(JSONL_PATH, "w", encoding="utf-8") as f_jsonl:

        csv_writer = csv.writer(f_csv)
        csv_writer.writerow([
            "index_name", "chunk_size", "chunk_overlap", "k",
            "question_id", "latency_total_sec", "latency_retrieval_sec",
            "latency_generation_sec", "retrieval_count"
        ])

        for k in K_VALUES:
            print(f"\n[INFO] Evaluating with k={k}...")
            retriever = vs.as_retriever(search_kwargs={"k": k})

            for i, q in enumerate(QUESTIONS, start=1):
                print(f"  [INFO] Question {i}/{len(QUESTIONS)}: {q['question']}")
                t0 = time.perf_counter()
                docs = retriever.get_relevant_documents(q["question"])
                t1 = time.perf_counter()
                retrieval_latency = t1 - t0

                context = build_context(docs)
                print(f"    [DEBUG] Retrieved {len(docs)} docs | Context length: {len(context)} chars")

                t2 = time.perf_counter()
                ans_msg = chain.invoke({"question": q["question"], "context": context})
                answer = getattr(ans_msg, "content", str(ans_msg)).strip()
                t3 = time.perf_counter()
                generation_latency = t3 - t2

                total_latency = t3 - t0

                csv_writer.writerow([
                    INDEX_DIR.name, cs, co, k, q["id"],
                    f"{total_latency:.3f}", f"{retrieval_latency:.3f}",
                    f"{generation_latency:.3f}", len(docs)
                ])

                detail = {
                    "index_name": INDEX_DIR.name,
                    "chunk_size": cs,
                    "chunk_overlap": co,
                    "k": k,
                    "question_id": q["id"],
                    "question": q["question"],
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
    print("[INFO] Evaluation completed successfully.")

if __name__ == "__main__":
    main()
