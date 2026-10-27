# Local PDF RAG

This starter extracts PDF text, makes paragraph-aware chunks, embeds them
locally with Sentence Transformers, stores text and vectors in SQLite, retrieves
matching chunks with cosine similarity, and can answer questions using a local
Ollama model.

## Local setup (Windows PowerShell)

Use Python 3.10 or newer. From this folder, create and activate a virtual
environment, then install the listed packages:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
conda install -c conda-forge "pymupdf>=1.24,<2" "sentence-transformers>=3,<7" "numpy>=1.26,<3"
```

On the first `add`, Sentence Transformers downloads the selected model to its
local cache. That first download needs internet access. Embedding and search
then run on your computer; set the device to `"cpu"` in `rag.py` to force CPU.

## Add, list, and search

```powershell
python rag.py preview .\2405.05672v1.pdf --page 2
python rag.py add .\2405.05672v1.pdf
python rag.py list
python rag.py search "How does the paper represent sign language gestures?"
python rag.py search "keypoint streams" --top-k 8
```

The database is created at `rag_data/rag.sqlite3`. It contains document
metadata, chunk text, source page/section, and each embedding. Back up that file
to preserve the retrieval index. To rebuild from scratch, remove the
`rag_data` folder and add the PDFs again.

## Change the defaults

All starting values are near the top of `rag.py`, under **Beginner-adjustable
settings**:

| Setting | Default | What it changes |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model used for both PDF chunks and search queries. If you change it after adding PDFs, re-add those PDFs so every stored vector uses the same model. |
| `EMBEDDING_DEVICE` | `None` | `None` lets the library choose; use `"cpu"` to force CPU or `"cuda"` for a configured CUDA environment. |
| `CHUNK_TARGET_WORDS` | `200` | Soft target size. Smaller chunks are more specific; larger chunks carry more context. |
| `CHUNK_MAX_WORDS` | `220` | Upper limit for a paragraph or split sentence piece. A long paragraph is split at sentence boundaries where possible. |
| `CHUNK_OVERLAP_WORDS` | `40` | Words copied from the previous chunk to preserve context across a boundary. Set to `0` for no overlap. |
| `EMBEDDING_BATCH_SIZE` | `32` | Number of chunks encoded together. Lower it if embedding runs out of memory. |
| `DEFAULT_TOP_K` | `6` | Number of results shown when `--top-k` is omitted. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server address. |
| `OLLAMA_MODEL` | `llama4:maverick` | Default answer model; use the --model option to select another installed model. |
| `OLLAMA_TIMEOUT_SECONDS` | `300` | Maximum wait for a response from Ollama. |
| `DATA_DIR` | `rag_data` beside `rag.py` | Where the SQLite index is stored. |
| `FULL_WIDTH_BLOCK_RATIO` | `0.62` | Layout heuristic for recognizing full-width PDF blocks before ordering columns. |

Use `preview` first when adjusting chunk settings. It runs extraction and
chunking only, so it does not load or download the embedding model and does not
change the saved index. For example, `python rag.py preview .\paper.pdf --page 2`
shows the chunks assigned to PDF page 2.

Chunk sizes are measured in words, while transformer limits are measured in
tokens. Word count is an approachable first approximation, not an exact token
limit. Review the first few chunks and adjust `CHUNK_TARGET_WORDS` and
`CHUNK_MAX_WORDS` before indexing a large collection.

## Ask questions with local Ollama

Make sure Ollama is running on this workstation and the selected model is
installed. The current default, llama4:maverick, is the model listed on the
lab4 workstation. Change OLLAMA_MODEL near the top of rag.py or pass another
installed model with the --model option.

    python rag.py ask "Which datasets were used for evaluation?"
    python rag.py ask "Which datasets were used for evaluation?" --top-k 10
    python rag.py ask "Which datasets were used for evaluation?" --model another-model

The ask command uses the same dense retrieval as search, then sends only the
retrieved chunks to Ollama through its local chat API. The prompt asks the
model to rely on those excerpts, cite claims with labels such as [S1], and say when the
retrieved evidence is insufficient. A source list maps each citation label to
the PDF file, page, section, and chunk ID. An answer can still be missed if the
retriever does not return the needed passage; ask does not yet add keyword or
iterative retrieval.

## What this version deliberately supports

- Text-based PDFs that contain selectable text. OCR for scanned PDFs is not
  included yet.
- Page-aware, section-aware chunks and result citations using PDF page numbers.
- Local dense embeddings and exact cosine similarity. The exact search scans
  every stored vector, which is simple and suitable for a small learning corpus.
- A basic column-order heuristic based on text-block positions. Complex tables,
  sidebars, and multi-column layouts still deserve visual inspection.

The `search` command remains available for inspecting the retrieved chunks
without generating an answer.
