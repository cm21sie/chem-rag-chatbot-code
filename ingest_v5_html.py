import os
import json
import csv
import re
from pathlib import Path
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# -----------------------
# CONFIG
# -----------------------
overview_file = Path("moduleoverviews.txt")
html_folder = Path("module_htmls")
chunk_size = 500
chunk_overlap = 50

id_map_file = 'id_map.csv'
chunks_file = 'chunks_with_metadata.json'
faiss_index_file = 'html_faiss_index.index'
faiss_metadata_schema_file = 'html_faiss_metadata_schema.json'

# -----------------------
# SETUP
# -----------------------
id_map = []
chunks_with_metadata = []
doc_id_counter = 0

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""]
)

# -----------------------
# PROCESS moduleoverviews.txt
# -----------------------
if overview_file.exists():
    with open(overview_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    current_module = "N/A"
    current_url = "N/A"

    for line in lines:
        module_match = re.match(r"-?\s*(CHEM\d{4})", line)
        if module_match:
            current_module = module_match.group(1)
            current_url = "N/A"
            id_map.append({
                "doc_id": f"doc_{doc_id_counter:04d}",
                "source_file": overview_file.name,
                "module_code": current_module,
                "url": current_url
            })
            doc_id_counter += 1

        url_match = re.search(r"(https?://\S+)", line)
        if url_match:
            current_url = url_match.group(1)
            if id_map:
                id_map[-1]["url"] = current_url

        chunks = text_splitter.split_text(line)
        for j, chunk in enumerate(chunks):
            chunks_with_metadata.append({
                "doc_id": f"doc_{doc_id_counter-1:04d}" if id_map else "doc_0000",
                "chunk_id": j,
                "module_code": current_module,
                "source_file": overview_file.name,
                "url": current_url,
                "text": chunk
            })
else:
    print(f"Warning: File not found -> {overview_file.name}")

# -----------------------
# PROCESS module_htmls/*.html
# -----------------------
if html_folder.exists():
    html_files = list(html_folder.glob("*.html"))
    print(f"Found {len(html_files)} HTML files.")

    for html_file in html_files:
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                text = soup.get_text(separator="\n").strip()

                # Try to extract module code
                module_match = re.search(r"(CHEM\d{4})", html_file.stem)
                content_module_match = re.search(r"(CHEM\d{4})", text)

                module_code = (
                    module_match.group(1)
                    if module_match
                    else content_module_match.group(1)
                    if content_module_match
                    else "N/A"
                )

                doc_id = f"doc_{doc_id_counter:04d}"
                id_map.append({
                    "doc_id": doc_id,
                    "source_file": html_file.name,
                    "module_code": module_code,
                    "url": "N/A"
                })
                doc_id_counter += 1

                chunks = text_splitter.split_text(text)
                for j, chunk in enumerate(chunks):
                    chunks_with_metadata.append({
                        "doc_id": doc_id,
                        "chunk_id": j,
                        "module_code": module_code,
                        "source_file": html_file.name,
                        "url": "N/A",
                        "text": chunk
                    })

        except Exception as e:
            print(f"Error reading {html_file.name}: {e}")
else:
    print(f"Warning: Folder not found -> {html_folder.resolve()}")

# -----------------------
# WRITE ID MAP AND CHUNKS
# -----------------------
with open(id_map_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["doc_id", "source_file", "module_code", "url"])
    writer.writeheader()
    for row in id_map:
        writer.writerow(row)

with open(chunks_file, 'w', encoding='utf-8') as f:
    json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)

print(f"Created {id_map_file} with {len(id_map)} entries")
print(f"Created {chunks_file} with {len(chunks_with_metadata)} chunks")

# -----------------------
# EMBEDDINGS + FAISS (OLLAMA)
# -----------------------
print("Creating embeddings with Ollama (model: nomic-embed-text)...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

documents = [
    Document(
        page_content=chunk["text"],
        metadata={
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
            "module_code": chunk["module_code"],
            "source_file": chunk["source_file"],
            "url": chunk["url"]
        }
    )
    for chunk in chunks_with_metadata
]

faiss_index = FAISS.from_documents(documents, embeddings)

faiss_index.save_local(faiss_index_file)

with open(faiss_metadata_schema_file, 'w', encoding='utf-8') as f:
    json.dump(["doc_id", "chunk_id", "module_code", "source_file", "url"], f, indent=2)

print(f"FAISS index saved to {faiss_index_file}")
print(f"Metadata schema saved to {faiss_metadata_schema_file}")
