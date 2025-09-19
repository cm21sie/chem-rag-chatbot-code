import re
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

# === FILE PATHS ===
overviews_path = Path("moduleoverviews.txt")
html_folder = Path("module_htmls")

# === READ FILES ===
file_contents = {}

# Read moduleoverviews.txt
if overviews_path.exists():
    with open(overviews_path, "r", encoding="utf-8") as f:
        file_contents[overviews_path.name] = f.read()
else:
    print(f"Warning: File not found -> {overviews_path.name}")
    file_contents[overviews_path.name] = ""

# Read HTML files in module_htmls folder
if html_folder.exists():
    html_files = list(html_folder.glob("*.html"))
    print(f"Found {len(html_files)} HTML files in '{html_folder.name}'.\n")

    for html_file in html_files:
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                text = soup.get_text(separator="\n").strip()
                file_contents[html_file.name] = text
        except Exception as e:
            print(f"Error reading {html_file.name}: {e}")
            file_contents[html_file.name] = ""
else:
    print(f"Warning: Folder not found -> {html_folder.resolve()}")

# === CLEANING FUNCTION ===
def clean_text(text):
    """Clean text: strip whitespace, normalize spaces, remove empty lines, deduplicate."""
    original_lines = text.splitlines()
    cleaned_lines = []

    for line in original_lines:
        line = line.strip()
        line = re.sub(r"\s+", " ", line)
        if line:
            cleaned_lines.append(line)

    seen = set()
    deduped_lines = []
    for line in cleaned_lines:
        if line not in seen:
            deduped_lines.append(line)
            seen.add(line)

    return deduped_lines, original_lines

# === LOGGING FUNCTION ===
def log_cleaning(file_name, original_lines, cleaned_lines, log_out):
    log_out.write(f"File: {file_name}\n")
    log_out.write(f"Original lines: {len(original_lines)}\n")
    log_out.write(f"Cleaned lines: {len(cleaned_lines)}\n")
    log_out.write("Sample changes (up to 5):\n")

    changes_shown = 0
    removed_lines = [line for line in original_lines if line.strip() and line.strip() not in cleaned_lines]
    for line in removed_lines[:5]:
        log_out.write(f"- Before: {repr(line)}\n")
        log_out.write(f"  After:  <removed>\n")
        changes_shown += 1

    for line in original_lines:
        clean = line.strip()
        clean = re.sub(r"\s+", " ", clean)
        if clean != line and clean in cleaned_lines and changes_shown < 5:
            log_out.write(f"- Before: {repr(line)}\n")
            log_out.write(f"  After:  {repr(clean)}\n")
            changes_shown += 1

    if changes_shown == 0:
        log_out.write("No changes detected.\n")

    log_out.write("\n" + "-" * 50 + "\n\n")

# === PROCESS FILES ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cleaning_log_filename = f"cleaning_log_{timestamp}.txt"
structured_output_filename = "cleaned_module_data.txt"

with open(cleaning_log_filename, "w", encoding="utf-8") as log_out, \
     open(structured_output_filename, "w", encoding="utf-8") as data_out:

    log_out.write(f"Cleaning Log - {timestamp}\n")
    log_out.write("=" * 60 + "\n\n")

    for file_name, content in file_contents.items():
        cleaned_lines, original_lines = clean_text(content)

        log_cleaning(file_name, original_lines, cleaned_lines, log_out)

        data_out.write(f"# {file_name}\n")
        for line in cleaned_lines:
            data_out.write(line + "\n")
        data_out.write("\n")

print(f"Cleaning log saved to: {cleaning_log_filename}")
print(f"Structured cleaned data saved to: {structured_output_filename}")
