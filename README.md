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
python rag.py search "keypoint streams" --top-k 6
python rag.py search "Which datasets were used?" --document "5563_Uni_Sign_Toward_Unified_S.pdf"
python rag.py search "Which datasets were used?" --document "5563_Uni_Sign_Toward_Unified_S.pdf" --retrieval-mode hybrid
python rag.py search "Which datasets were used?" --document "5563_Uni_Sign_Toward_Unified_S.pdf" --retrieval-mode rerank
python rag.py search "Which datasets were used?" --document "5563_Uni_Sign_Toward_Unified_S.pdf" --retrieval-mode hybrid-rerank
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
| `CHUNK_TARGET_WORDS` | `80` | Soft target size. Smaller chunks are more specific; larger chunks carry more context. |
| `CHUNK_MAX_WORDS` | `100` | Upper limit for a paragraph or split sentence piece. A long paragraph is split at sentence boundaries where possible. |
| `CHUNK_OVERLAP_WORDS` | `15` | Words copied from the previous chunk to preserve context across a boundary. Set to `0` for no overlap. |
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
`CHUNK_MAX_WORDS` before indexing a large collection. After changing chunk settings, run the add command again for each PDF to rebuild its stored chunks and embeddings.

## Ask questions with local Ollama

Make sure Ollama is running on this workstation and the selected model is
installed. The current default, llama4:maverick, is the model listed on the
lab4 workstation. Change OLLAMA_MODEL near the top of rag.py or pass another
installed model with the --model option.

    python rag.py ask "Which datasets were used for evaluation?"
    python rag.py ask "Which datasets were used for evaluation?" --top-k 6
    python rag.py ask "Which datasets were used for evaluation?" --document "5563_Uni_Sign_Toward_Unified_S.pdf" --model qwen3:8b
    python rag.py ask "Which datasets were used for evaluation?" --document "5563_Uni_Sign_Toward_Unified_S.pdf" --retrieval-mode hybrid
    python rag.py ask "Which datasets were used for evaluation?" --document "5563_Uni_Sign_Toward_Unified_S.pdf" --retrieval-mode hybrid-rerank

The ask command uses the selected retrieval mode, then sends only the retrieved
chunks to Ollama through its local chat API. Dense cosine retrieval remains the
default. The prompt asks the model to rely on those excerpts, cite claims with
labels such as [S1], and say when the retrieved evidence is insufficient. A
source list maps each citation label to the PDF file, page, section, and chunk
ID. Only sources cited in the generated answer are printed. If only one PDF is
indexed, ask uses it automatically. If several PDFs are indexed, specify one with --document;
this prevents the answer from combining different papers. Search still searches
all PDFs unless --document is supplied. An answer can still be missed if the
retriever does not return the needed passage. Dense search is the baseline.
Add --retrieval-mode hybrid to search or ask to combine dense rankings with BM25
keyword rankings using reciprocal rank fusion. Keyword matching rewards exact
word overlap, while dense matching can find semantically related wording.
Hybrid mode uses the same SQLite index and requires no re-indexing.
Its score is a fused ranking score, so compare chunk IDs and ranks rather than
numeric scores between dense and hybrid results. By default, hybrid mode
considers up to max(30, top-k times 5) candidates from each ranking. The
candidate count and reciprocal-rank constant are adjustable near the top of
rag.py.

Rerank mode first selects 30 dense candidates; hybrid-rerank selects candidates
with hybrid retrieval. Both then score the question and each passage together
with a local CrossEncoder and return the top-k results. The reranker model is
downloaded on first use and cached locally; dense and hybrid modes do not load
it. RERANK_CANDIDATES, RERANK_BATCH_SIZE, and RERANK_MAX_LENGTH are adjustable
near the top of rag.py. Reranking does not require re-indexing the PDFs.

## Evaluate retrieval against your benchmark

Create a CSV from your benchmark questions using `benchmark_template.csv`. Use
one row per question and these columns:

- `question_id`: a stable short label such as `q01`.
- `document`: exact indexed PDF filename; leave blank to search all indexed PDFs.
- `question`: the search question.
- `answerability`: `answerable`, `unanswerable`, or `uncertain`.
- `relevant_chunk_ids`: **all** chunks you have judged relevant to answering the
  question, separated by `|`. If a question needs two passages, record both IDs.
  Add newly identified relevant chunks here too, even if an earlier run did not
  retrieve them.

For an answerable question, provide at least one relevant chunk ID. If you have
checked the selected PDF and it contains no evidence for the question, mark it
`unanswerable` and leave `relevant_chunk_ids` blank. If you have not established
whether the PDF contains evidence, mark it `uncertain`; do not treat missing
labels as proof that the paper lacks the answer. Unanswerable and uncertain
questions still have their retrieved chunks saved, but they are excluded from
relevance ranking metrics.

Example `relevant_chunk_ids` value: `248cce977151-c0007|248cce977151-c0011`.
The IDs must exist in the current SQLite index. If chunking settings changed,
re-add the PDF and update labels to the new chunk IDs before evaluating.

Run the benchmark with:

```powershell
python evaluate.py benchmark.csv --retrieval-depth 10 --metric-k 6
```

The evaluator compares dense, hybrid, rerank, and hybrid-rerank by default. The
reranker may download its model on first use. To run only the two fast baseline
modes, add `--modes dense hybrid`. This evaluates retrieval only and does not
call Ollama.

It writes three files under `evaluation_results/`:

- `retrieved_chunks.csv`: every returned chunk through rank 10, with its text,
  score, rank, and whether its ID is marked relevant.
- `question_metrics.csv`: per-question answerability, Recall@6, Precision@6,
  MRR@6, first relevant rank, complete coverage, and any labeled chunks not
  retrieved. Ranking metrics are blank for unanswerable and uncertain questions.
- `mode_summary.csv`: macro averages and success/complete-coverage rates by
  retrieval mode.

`--retrieval-depth` controls how many ranked chunks are saved; `--metric-k`
controls the cutoff scored. The defaults save the top 10 while scoring the
agreed top 6. Recall@6 is the fraction of all labeled relevant chunks found in
the first 6. Complete coverage means every labeled relevant chunk for that
question appeared in the first 6. These metrics depend on labeling all relevant
evidence chunks, so include more than just the chunk returned by one particular
run.

## What this version deliberately supports

- Text-based PDFs that contain selectable text. OCR for scanned PDFs is not
  included yet.
- Page-aware, section-aware chunks and result citations using PDF page numbers.
- Local dense embeddings and exact cosine similarity. The exact search scans
  every stored vector, which is simple and suitable for a small learning corpus.
- Optional local BM25 keyword retrieval and CrossEncoder reranking. Dense-only
  remains the default so each retrieval mode can be compared against the benchmark.
- A basic column-order heuristic based on text-block positions. Complex tables,
  sidebars, and multi-column layouts still deserve visual inspection.

The `search` command remains available for inspecting the retrieved chunks
without generating an answer.
