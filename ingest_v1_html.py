from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

# === FOLDER PATH FOR HTML FILES ===
html_folder = Path("module_htmls")
module_overviews_file = Path("moduleoverviews.txt")

# === READ HTML FILES AND STORE CONTENTS ===
file_contents = {}

# Read HTML files
if html_folder.exists():
    html_files = list(html_folder.glob("*.html"))
    print(f"Found {len(html_files)} HTML files in '{html_folder.name}'.\n")

    for html_file in html_files:
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                text = soup.get_text(separator="\n").strip()
                file_contents[html_file.name] = text
                print(f"Read {len(text)} characters from {html_file.name}")
        except Exception as e:
            print(f"Error reading {html_file.name}: {e}")
            file_contents[html_file.name] = ""
else:
    print(f"Folder not found: {html_folder.resolve()}")

# Read moduleoverviews.txt
if module_overviews_file.exists():
    try:
        with open(module_overviews_file, "r", encoding="utf-8") as f:
            overview_text = f.read().strip()
            file_contents[module_overviews_file.name] = overview_text
            print(f"\nRead {len(overview_text)} characters from {module_overviews_file.name}")
    except Exception as e:
        print(f"\nError reading {module_overviews_file.name}: {e}")
        file_contents[module_overviews_file.name] = ""
else:
    print(f"\nFile not found: {module_overviews_file.resolve()}")

# === PRINT BASIC INFO TO CONSOLE ===
for file_name, content in file_contents.items():
    print(f"\nFile: {file_name}")
    print(f"Total characters: {len(content)}")
    print(f"First 500 characters:\n{content[:500]}")

# === SAVE SMOKE TEST OUTPUT TO FILE ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"ingestion_test_output_{timestamp}.txt"

with open(output_filename, "w", encoding="utf-8") as out:
    out.write(f"Ingestion Test Output - {timestamp}\n")
    out.write("=" * 50 + "\n\n")
    for file_name, content in file_contents.items():
        out.write(f"File: {file_name}\n")
        out.write(f"Total characters: {len(content)}\n")
        out.write("First 500 characters:\n")
        out.write(content[:500].replace("\n", "\\n") + "\n\n")

print(f"\nSmoke test file saved to: {output_filename}")
