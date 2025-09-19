import re
import time
import csv
import json
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings

CSV_OUTPUT_FILE = "html_baseline_results.csv"
JSON_EXAMPLE_FILE = "html_example_baseline_run.json"

INDEX_PATH = "html_faiss_index.index"  # new FAISS index name
EMBED_MODEL = "nomic-embed-text"       # must match embedding used to build index
LLM_MODEL = "llama3"                   # change to test other models
TOP_K = 6

# --- Prompt Template ---
system_prompt = SystemMessagePromptTemplate.from_template(
    """You are a helpful assistant answering questions using ONLY the provided context.
If the answer is not in the context, reply "I don't know"."""
)
human_prompt_template = HumanMessagePromptTemplate.from_template(
    "Context:\n{context}\n\nQuestion: {question}"
)
prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt_template])

# --- Embeddings with Ollama ---
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# --- Load FAISS vectorstore (new index path) ---
vectorstore = FAISS.load_local(
    INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# Retriever (use invoke API; MMR is a bit more robust for diverse context)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": TOP_K, "fetch_k": 40, "lambda_mult": 0.5}
)

# --- Load Ollama LLM ---
llm = OllamaLLM(model=LLM_MODEL, temperature=0)

# --- RAG Chain ---
rag_chain = prompt | llm | StrOutputParser()

QUESTIONS = [
    "What is the credit value of CHEM2112?",
    "How many second-year modules are available?",
    "Are there any first-year modules assessed by 100% coursework?",
    "Summarise what topics are covered across all first-year chemistry modules.",
    "Are there any modules specifically about nanotechnology?",
    "How do you know which modules are second-year modules?"
]

results = []

for i, question in enumerate(QUESTIONS):
    print(f"Processing question {i+1}/{len(QUESTIONS)}...")

    # Retrieve (invoke, not deprecated get_relevant_documents)
    docs = retriever.invoke(question)
    # Build context
    context = "\n".join(d.page_content for d in docs)
    # Optional: capture brief source info for debugging/comparison
    sources = [f"{d.metadata.get('source_file','')}|{d.metadata.get('module_code','')}" for d in docs]

    # Generate answer
    start_time = time.time()
    answer = rag_chain.invoke({"context": context, "question": question})
    latency = time.time() - start_time

    # Save row
    results.append({
        "question": question,
        "retrieved_chunks": context,
        "answer": answer,
        "latency": round(latency, 3),
        "sources": " ; ".join(sources)
    })

# --------------------------
# SAVE CSV
# --------------------------
with open(CSV_OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "retrieved_chunks", "answer", "latency", "sources"])
    writer.writeheader()
    writer.writerows(results)

print(f"Baseline results saved to {CSV_OUTPUT_FILE}")

# --------------------------
# SAVE ONE EXAMPLE JSON
# --------------------------
if results:
    example = results[0]
    example_json = {
        "question": example["question"],
        "retrieved_chunks": example["retrieved_chunks"],
        "answer_cleaned": example["answer"],
        "sources": example["sources"]
    }
    with open(JSON_EXAMPLE_FILE, "w", encoding="utf-8") as f:
        json.dump(example_json, f, indent=2)
    print(f"Example run saved to {JSON_EXAMPLE_FILE}")
