# ingest_indices.py
from pathlib import Path
import json
import re
from datetime import datetime
from typing import List, Dict, Any

from langchain_community.document_loaders import BSHTMLLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# ========= SETTINGS =========
HTML_FOLDER = Path("module_htmls")
OVERVIEW_FILE = Path("moduleoverviews.txt")
INDEX_ROOT = Path("chatbot_project")
INDEX_ROOT.mkdir(exist_ok=True)

# Tweak these as needed
CHUNK_SIZES = [200, 500, 1000]
CHUNK_OVERLAPS = [20, 50, 150]

# Text splitter config (stable across all runs for fairness)
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,  # placeholder; will be overridden per run
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
    length_function=len,
)

# Embeddings (local via Ollama)
EMBEDDINGS = OllamaEmbeddings(model="nomic-embed-text")

# ========= HELPERS =========
def normalise_whitespace(s: str) -> str:
    # collapse multiple whitespace and strip
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def load_all_docs() -> List[Any]:
    docs = []
    # HTML docs
    html_files = sorted(HTML_FOLDER.glob("*.html"))
    for hf in html_files:
        try:
            loader = BSHTMLLoader(str(hf))
            loaded = loader.load()
            for d in loaded:
                d.page_content = normalise_whitespace(d.page_content)
                d.metadata = {**d.metadata, "source_path": str(hf), "source_type": "html"}
            docs.extend(loaded)
        except Exception as e:
            print(f"[WARN] Failed to read {hf.name}: {e}")

    # moduleoverviews.txt (optional but expected)
    if OVERVIEW_FILE.exists():
        try:
            tloader = TextLoader(str(OVERVIEW_FILE), encoding="utf-8")
            loaded = tloader.load()
            for d in loaded:
                d.page_content = normalise_whitespace(d.page_content)
                d.metadata = {**d.metadata, "source_path": str(OVERVIEW_FILE), "source_type": "txt"}
            docs.extend(loaded)
        except Exception as e:
            print(f"[WARN] Failed to read {OVERVIEW_FILE.name}: {e}")
    else:
        print(f"[INFO] {OVERVIEW_FILE} not found; continuing with HTML only.")

    print(f"[INFO] Loaded {len(docs)} raw documents.")
    return docs

def split_docs(raw_docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    splits = splitter.split_documents(raw_docs)
    # add chunk_size/overlap into metadata for transparency
    for i, d in enumerate(splits):
        d.metadata = {
            **d.metadata,
            "chunk_id": i,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
    return splits

def save_manifest(index_dir: Path, manifest: Dict[str, Any]):
    with open(index_dir / "build_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

# ========= MAIN =========
def main():
    raw_docs = load_all_docs()
    if not raw_docs:
        print("[ERROR] No documents found. Check your paths.")
        return

    for cs in CHUNK_SIZES:
        for co in CHUNK_OVERLAPS:
            index_dir = INDEX_ROOT / f"faiss_cs{cs}_co{co}"
            index_dir.mkdir(exist_ok=True)
            print(f"\n[BUILD] chunk_size={cs}, chunk_overlap={co}")

            chunks = split_docs(raw_docs, chunk_size=cs, chunk_overlap=co)
            print(f"[INFO] {len(chunks)} chunks created. Building FAISS index...")

            vs = FAISS.from_documents(chunks, EMBEDDINGS)
            vs.save_local(str(index_dir))  # will create index.faiss / index.pkl

            manifest = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "chunk_size": cs,
                "chunk_overlap": co,
                "num_chunks": len(chunks),
                "sources": list(sorted({d.metadata.get("source_path", "unknown") for d in chunks})),
                "notes": "Built with OllamaEmbeddings(nomic-embed-text) and RecursiveCharacterTextSplitter",
            }
            save_manifest(index_dir, manifest)
            print(f"[OK] Saved index to {index_dir} with {manifest['num_chunks']} chunks.")

if __name__ == "__main__":
    main()
