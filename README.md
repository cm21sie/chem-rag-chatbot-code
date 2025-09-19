# chem-rag-chatbot-code
Python scripts for ingestion, retrieval, evaluation, and deployment of a local RAG-based chatbot for the University of Leeds Chemistry department.

ingest_v1.py - Performs a smoke test of raw HTML ingestion. Reads all module HTML files and ‘moduleoverviews.txt’. Extracts plain text using BeautifulSoup. Prints basic file stats to the console (file name, character count, first 500 characters). Saves results to a text file for verification.

ingest_v1_output.txt - Stores a timestamped smoke test report containing the filename, character count, and first 500 characters of text extracted from each HTML module file and ‘moduleoverviews.txt’. Used to verify that the raw files were successfully read and contain meaningful data before further processing.

ingest_v2.py - Performs HTML cleaning and structuring. Removes navigation elements, repeated headers/footers, boilerplate text, and irrelevant formatting from the raw HTML extracted in ‘ingest_v1.py’. Produces a cleaned dataset ready for chunking and embedding. Generates a detailed cleaning log.

ingest_v2_cleaning_log.txt - Shows original vs. cleaned line counts and up to five before/after examples per file.

ingest_v2_cleaned_module_data.txt - Consolidated, cleaned corpus ready for chunking/embedding.

ingest_v3.py - Chunks the corpus for retrieval. Reads ‘moduleoverviews.txt’ and all module html documents. Extracts plain text. Concentrates with per-file headers. Splits into overlapping chunks using ‘RecursiveCharacterTextSplitter’. Writes a small JSON preview of the first N chunks with module code/URL hints.

ingest_v3_chunked_data_preview.json - Small sample of the first 10 chunks with snippet, inferred module code, and URL. Used to verify chunk size/overlap and provenance cues before full indexing.

ingest_v4.py - Parses all HTML and overview files. Assigns unique document IDs. Splits content into overlapping text chunks. Attaches metadata (module code, source, URL).

id_map.csv - A lookup table assigning each processed document a unique ‘doc_id’, alongside its source filename, detected module code, and associated URL. Provides traceability between chunks and their original source.

chunks_with_metadata.json - Contains every text chunk created from the input documents, each annotated with metadata (‘doc_id’, ‘chunk_id’, module code, source file, URL, and text).

ingest_v5.py - Builds the searchable vector index. Embeds all chunks with Ollama’s ‘nomic-embed-text’. Creates a FAISS index and a small metadata-schema manifest.

faiss_metaschema_schema.json - Minimal schema/field list describing per-document metadata stored alongside embeddings (doc_id, chunk_id, module_code, source_file, url).

retrieval.py - Retrieval smoke test. Loads FAISS + embeddings. Runs MMR retrieval. Boosts by module code when present. Attempts direct regex extraction of credit lines. Falls back to LLM if needed.

baseline_results.py - Baseline RAG evaluation. Uses a fixed prompt, FAISS retriever (MMR) and llama3 to answer a small suite of questions. Records latency and sources.

baseline_results.csv - Summary of questions with retrieved context, answer, total/phase latencies, and source hints.

example_baseline_run.json - Single example (first question) with full retrieved context and answer for manual inspection.

deepseek_model_evaluation.py - Model-specific evaluation. Same RAG flow as baseline but using the DeepSeek model.

deepseek_model_results.csv - Summary CSV of Q/A/latencies for the DeepSeek run.

deepseek_model_output_examples.json - JSON with example outputs and top retrieved docs (doc/chunk/source traceability).

llama3_model_evaluation.py - Model-specific evaluation. Same RAG flow as baseline but using the LLaMA 3 model.

llama3_model_results.csv - Summary CSV of Q/A/latencies for the LLaMA 3 run.

llama3_model_output_examples.json - JSON with example outputs and top retrieved docs (doc/chunk/source traceability).

gemma3_model_evaluation.py - Model-specific evaluation. Same RAG flow as baseline but using the Gemma 3 model.

gemma3_model_results.csv - Summary CSV of Q/A/latencies for the Gemma 3 run.

gemma3_model_output_examples.json - JSON with example outputs and top retrieved docs (doc/chunk/source traceability).

overlap_index_creation.py - Builds multiple FAISS indexes across chunk size/overlap grid. Loads HTML and overview documents, normalizes whitespace, splits with requested chunk_size/chunk_overlap, embeds with nomic-embed-text, and saves indexes.

chunking_eval_details.json - JSONL of per-question details (retrieved docs, snippets, scores, latencies) for each index condition.

chunking_eval.csv - Summary CSV of scoring/latency across chunk size/overlap conditions.

k_test.py - k sweep evaluation for a single index. Measures retrieval/generation/total latency and captures answers as k varies.

faiss_cs500_co50_k_eval_details.json - JSONL of detailed per-question results for each k (retrieved docs, snippets, timings, answer).

faiss_cs500_co50_k_eval.csv - Summary CSV across k values with latency breakdowns and retrieval counts.

prompt_evaluation.py - Prompt template comparison at fixed k. Evaluates ‘structured, ‘counting + summarisation’ and ‘bullet points’ prompts over the same questions. Logs latency and retrieved context length.

prompt_eval_k8_details.json - JSONL with per-template, per-question details (answer, latencies, retrieved doc/snippets).

prompt_eval_k8.csv - Summary CSV of per-template performance at k=8 (total/retriever/generation latency and retrieval counts).

streamlit_deployment_v1.py - Baseline deployment - no caching, extra features, or tabs.

streamlit_deployment_v2.py - Adds caching (for embeddings, retriever, and chain). Adds multiple tabs (Chat, Sources, About). Adds toggles to show retrieved context and sources after each output.

streamlit_deployment_v3.py - Adds export features (download answers and copy to clipboard). Adds feedback buttons (thumbs up/down) after each output.

streamlit_deployment_v4.py - Adds progress feedback (stepwise status + progress bar) during retrieval and generation.

streamlit_deployment_v5.py - Adds a sidebar disclaimer and data/privacy notes (usage, retention, and safety guidance).
