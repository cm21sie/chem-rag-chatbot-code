import json
import re
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === FILE PATHS ===
html_folder = Path("module_htmls")
overviews_file = Path("moduleoverviews.txt")
chunk_preview_file = Path("chunked_data_preview.json")

# === PARAMETERS ===
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # characters overlap between chunks
MAX_PREVIEW_CHUNKS = 10  # how many chunks to show in the preview JSON

# === READ AND COMBINE RAW TEXT DATA ===
combined_text = ""

# Read moduleoverviews.txt
if overviews_file.exists():
    with open(overviews_file, "r", encoding="utf-8") as f:
        overview_text = f.read().strip()
        combined_text += f"# {overviews_file.name}\n{overview_text}\n\n"
else:
    print(f"Warning: File not found -> {overviews_file.name}")

# Read all HTML files in the folder
if html_folder.exists():
    html_files = list(html_folder.glob("*.html"))
    print(f"Found {len(html_files)} HTML files in '{html_folder.name}'.")

    for html_file in html_files:
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                text = soup.get_text(separator="\n").strip()
                combined_text += f"# {html_file.name}\n{text}\n\n"
        except Exception as e:
            print(f"Error reading {html_file.name}: {e}")
else:
    print(f"Warning: HTML folder not found -> {html_folder.resolve()}")

# === CHUNKING USING RecursiveCharacterTextSplitter ===
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len
)

chunks = splitter.split_text(combined_text)

# === PREPARE PREVIEW JSON WITH MODULE INFO ===
chunk_preview = []
for i, chunk in enumerate(chunks[:MAX_PREVIEW_CHUNKS], start=1):
    # Extract module code and URL from the chunk
    module_code_match = re.search(r'\b[A-Z]{4}\d{4}\b', chunk)
    url_match = re.search(r'https?://[^\s]+', chunk)

    module_code = module_code_match.group(0) if module_code_match else "N/A"
    url = url_match.group(0) if url_match else "N/A"

    snippet_text = f"{chunk[:80]}... (Module: {module_code}, URL: {url})"

    chunk_preview.append({
        "chunk_id": i,
        "text_snippet": snippet_text
    })

# === SAVE PREVIEW JSON ===
with open(chunk_preview_file, "w", encoding="utf-8") as f:
    json.dump(chunk_preview, f, indent=4, ensure_ascii=False)

# === LOG INFO TO CONSOLE ===
print(f"Total chunks created: {len(chunks)}")
print(f"Chunk preview saved to: {chunk_preview_file}")
