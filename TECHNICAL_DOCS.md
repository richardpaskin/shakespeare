# Shakespeare RAG — Technical Documentation

A retrieval-augmented Q&A app for the complete works of Shakespeare. A local
LlamaIndex pipeline retrieves passages from a persistent ChromaDB collection
and a local Ollama LLM produces grounded answers with citations, served by a
Gradio chat UI.

## Architecture

```
              ┌────────────────────────────────────────────────┐
              │ Gradio ChatInterface  (Shakespeare.py)         │
              │  fn = answer(message, history)                 │
              └────────────────┬───────────────────────────────┘
                               │ user message + history
                               ▼
              ┌────────────────────────────────────────────────┐
              │ CondensePlusContextChatEngine                  │
              │  1. Condense step  ── Ollama LLM ──▶ rewritten │
              │     query (pronouns resolved, prior turn       │
              │     context inlined)                           │
              └────────────────┬───────────────────────────────┘
                               │ condensed query
                               ▼
              ┌────────────────────────────────────────────────┐
              │ PlayFilterRetriever                            │
              │  - detect_work_code(condensed)  →  work_code   │
              │     • PLAY_ALIASES (hand-written)              │
              │     • character_aliases.json (unique speakers) │
              │  - VectorStoreIndex.as_retriever(              │
              │       filters=work_code, top_k=TOP_K)          │
              └────────────────┬───────────────────────────────┘
                               │ top-K NodeWithScore
                               ▼
              ┌────────────────────────────────────────────────┐
              │ Chroma (persistent, on disk in chroma_db/)     │
              │  Embeddings via Ollama (mxbai-embed-large)     │
              │  Metadata: work_code, act, scene, file_name,…  │
              └────────────────┬───────────────────────────────┘
                               │ retrieved passages → context
                               ▼
              ┌────────────────────────────────────────────────┐
              │ Context step  ── Ollama LLM ──▶ streamed reply │
              │  tokens yielded to Gradio; metrics + sources   │
              │  written; references block appended            │
              └────────────────────────────────────────────────┘
```

Everything runs locally: **Ollama** serves both the embedding model and the
chat LLM over `localhost`; **Chroma** is an on-disk persistent client; **Gradio**
serves the UI on `127.0.0.1:7860`.

## Module map

| File | Role |
|---|---|
| [BardWorksSetup.py](BardWorksSetup.py) | One-off ingestion: parse `books/` HTML (or PDF) → embed → persist to `chroma_db/`. |
| [Shakespeare.py](Shakespeare.py) | The live app: load index, wire retriever + chat engine, serve Gradio UI, record metrics. |
| [build_character_aliases.py](build_character_aliases.py) | Offline build of `character_aliases.json` from speaker tags in `books/*.html`. |
| [analyse_queries.py](analyse_queries.py) | CLI report over `query_metrics.jsonl`. |
| [requirements.txt](requirements.txt) | Python dependency list. |
| `character_aliases.json` | ~760 character → `work_code` mappings, generated. |
| `chroma_db/` | Persistent vector store (sqlite + binary blobs). Gitignored. |
| `books/` | Source HTML, organised as `books/<work_code>/<work_code>.<act>.<scene>.html` plus `books/Poetry/*.html`. Gitignored. |
| `query_metrics.jsonl` | Per-query timing + routing rows written at runtime. Gitignored. |
| `llama_index.log` / `shakespeare_ui.log` | Runtime logs. Gitignored. |

## Data flow for a typical query

For the user message **"who does he fight?"** following a turn about Tybalt:

1. **Gradio** calls [`answer(message, history)`](Shakespeare.py#L232) as a streaming generator.
2. `answer` builds a fresh [`PlayFilterRetriever`](Shakespeare.py#L120) wrapping the persistent index, then a [`CondensePlusContextChatEngine`](Shakespeare.py#L237) with the replayed chat memory.
3. The chat engine's **condense step** calls the LLM to rewrite the question with prior context inlined → *"Who does Tybalt fight in the play Romeo and Juliet?"*.
4. The chat engine calls `PlayFilterRetriever._retrieve(query_bundle)` with the **condensed** string. [`detect_work_code`](Shakespeare.py#L106) runs on that string:
   - First checks `PLAY_ALIASES` (substring match, longest-first). "Romeo and Juliet" hits → `work_code = "romeo_juliet"`.
   - If no play name, falls back to the compiled character-name regex (word-boundary, longest-first). A bare "Tybalt" would also have routed to `romeo_juliet`.
5. The retriever builds a Chroma metadata filter `work_code == "romeo_juliet"` and asks the inner `VectorStoreIndex` retriever for the top-`TOP_K` nearest neighbours.
6. **Chroma** returns the top-K passages from documents tagged with that `work_code` only — guaranteeing the LLM sees Romeo & Juliet text even if a higher-scoring chunk from another play would otherwise have won.
7. The chat engine's **context step** stitches the passages into a system prompt and streams a reply from the LLM. `answer` yields tokens to Gradio as they arrive.
8. After the stream ends, `answer` writes one row to `query_metrics.jsonl` and yields a final string with the **References** block appended.

A single-turn message ("Tell me about Tybalt") follows the same path; the condense step typically passes it through unchanged, and the alias regex routes it directly.

## Module reference

### BardWorksSetup.py — ingestion

Run once (or with `--reset` to rebuild). Reads source files, parses them into
LlamaIndex `Document`s with rich metadata, embeds them via Ollama, and persists
to Chroma.

| Symbol | Purpose |
|---|---|
| `SCENE_FILENAME` ([:19](BardWorksSetup.py#L19)) | Regex `^(work).(act).(scene).html?$` — pulls `work_code`, act, scene from filenames like `macbeth.3.1.html`. |
| `BOILERPLATE_FRAGMENTS` ([:23](BardWorksSetup.py#L23)) | Lines containing these substrings are dropped during scrubbing ("Previous scene", etc.). |
| `load_pdf_documents` ([:32](BardWorksSetup.py#L32)) | PDF path: `SimpleDirectoryReader` over `*.pdf`, recursive. |
| `_clean_soup` ([:40](BardWorksSetup.py#L40)) | Removes `<script>/<style>/<meta>/<link>` and the nav `<table>` containing `td.nav`. |
| `_scrub_text` ([:50](BardWorksSetup.py#L50)) | Drops blank lines and any line containing a boilerplate fragment. |
| `_first_text_child` ([:57](BardWorksSetup.py#L57)) | Returns the first direct text child of a tag — used to extract the play name from `<td class="play">` without the malformed-HTML problem where the nav table is nested inside the play cell. |
| `_parse_play_html` ([:68](BardWorksSetup.py#L68)) | Builds a `Document` from a play scene HTML file. Metadata: `file_name`, `file_path`, `play`, `work_code`, `act`, `scene`, `work_type="play"`. Prepends a human-readable header like `From Romeo and Juliet, Act 3, Scene 1.` to the text body so the chunk carries its provenance even when retrieved standalone. |
| `_parse_poetry_html` ([:111](BardWorksSetup.py#L111)) | Same idea for the `Poetry/` subdirectory. Uses the `<title>` tag and sets `work_type="poetry"`. No act/scene. |
| `load_html_documents` ([:135](BardWorksSetup.py#L135)) | Walks `source_dir` for `*.html` / `*.htm`, dispatches each path to the poetry or play parser by whether `Poetry` is in its path. |
| `LOADERS` ([:146](BardWorksSetup.py#L146)) | Dispatch table `{"pdf": …, "html": …}`. Selected by `--filetype`. |
| `build_index` ([:152](BardWorksSetup.py#L152)) | Wires `OllamaEmbedding`, loads documents, optionally drops the collection (`--reset`), and persists via `VectorStoreIndex.from_documents`. |

**CLI flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--filetype {pdf,html}` | `pdf` | Which loader to use. The committed index was built from HTML. |
| `--source-dir PATH` | `./books` | Where to read from. |
| `--chroma-dir PATH` | `./chroma_db` | Where to persist. |
| `--reset` | off | Drop the collection first. Use this whenever you re-ingest, or you'll get duplicate embeddings. |

### Shakespeare.py — runtime app

#### Module-level state

| Symbol | Purpose |
|---|---|
| `PROJECT_DIR`, `CHROMA_DIR`, `METRICS_PATH` ([:22-24](Shakespeare.py#L22-L24)) | Paths derived from this file's location. |
| `log` ([:33](Shakespeare.py#L33)) | Named logger `shakespeare` writing to `llama_index.log`. Noisy HTTP/chromadb loggers are clamped to WARNING. |
| `COLLECTION_NAME`, `EMBED_MODEL`, `LLM_MODEL`, `TOP_K`, `KEEP_ALIVE` ([:35-39](Shakespeare.py#L35-L39)) | Wiring constants — see Configuration below. |
| `PLAY_ALIASES` ([:43](Shakespeare.py#L43)) | Hand-written `[(alias, work_code), …]` sorted longest-first so `"henry iv part 1"` matches before `"henry iv"`. |
| `_CHAR_PATTERN`, `_CHAR_LOOKUP` ([:103-104](Shakespeare.py#L103-L104)) | Compiled regex over every unique character name + lookup table back to `work_code`. Built at import. |
| `index` ([:170](Shakespeare.py#L170)) | The live `VectorStoreIndex`, loaded once at import. Followed immediately by a `Settings.llm.complete("Hello")` to warm Ollama. |

#### Functions

| Symbol | Purpose |
|---|---|
| `_load_character_pattern` ([:87](Shakespeare.py#L87)) | Reads `character_aliases.json` and compiles **one** regex `\b(name1|name2|…)\b` with names sorted longest-first. Python `\|` is first-match (not longest-match), so order matters when one name is a prefix of another (e.g. `"adriano de armado"` must precede `"adriano"`). |
| `detect_work_code(text)` ([:106](Shakespeare.py#L106)) | Two-stage routing: PLAY_ALIASES substring match wins; otherwise the character regex; otherwise `None` (no filter applied). |
| `PlayFilterRetriever` ([:120](Shakespeare.py#L120)) | A `BaseRetriever` subclass. `_retrieve` runs `detect_work_code` on `query_bundle.query_str` (which the chat engine has already **condensed**), builds a `MetadataFilters` on `work_code` if a play was identified, and delegates to a fresh inner retriever. Stashes `last_query` / `last_work_code` so the caller can read what the routing decided. |
| `load_index` ([:153](Shakespeare.py#L153)) | Configures `Settings.embed_model` and `Settings.llm`, opens the persistent Chroma collection, and returns `VectorStoreIndex.from_vector_store(...)`. |
| `_extract_text(obj)` ([:173](Shakespeare.py#L173)) | Gradio passes message content as `str`, `list` (multimodal parts), or `dict`. Recursively flattens any shape into a plain string. |
| `build_memory(history)` ([:188](Shakespeare.py#L188)) | Replays Gradio chat history into a `ChatMemoryBuffer(token_limit=3000)`. Strips the trailing `**References:**` block from prior assistant turns so the LLM doesn't learn to repeat citations. |
| `format_sources(response)` ([:210](Shakespeare.py#L210)) | Renders a markdown `**References:**` list: file name, page, score, and a 200-char snippet per source node. |
| `_record_metric(metric)` ([:224](Shakespeare.py#L224)) | Appends one JSON line to `query_metrics.jsonl`. Wrapped in `try/except` so a logging failure can't break the chat reply. |
| `answer(message, history)` ([:232](Shakespeare.py#L232)) | The Gradio entry point. **Generator**: yields the streamed reply incrementally, then yields one final string with the References block appended. Builds a per-request `PlayFilterRetriever` + `CondensePlusContextChatEngine`, logs retrieved nodes, times the run, and writes a metrics row containing: `ts`, `question`, `condensed_query`, `work_code`, `history_turns`, `setup_s`, `first_token_s`, `total_s`, `response_chars`, `num_sources`. |
| `CSS` ([:282](Shakespeare.py#L282)) | Borders + focus state for the chat input. |
| `demo` ([:293](Shakespeare.py#L293)) | The `gr.Blocks` UI hosting a `gr.ChatInterface(fn=answer, …)`. |

### build_character_aliases.py — alias dataset builder

Offline. Scans `books/<play>/*.html` for `<b>SPEAKER</b>` tags, keeps speakers
that appear in exactly one play, drops generic role names, and writes a flat
JSON mapping `{lowercased_name: work_code}` to `character_aliases.json`.

| Symbol | Purpose |
|---|---|
| `STOPLIST` ([:28](build_character_aliases.py#L28)) | Generic roles to discard even if unique-to-one-play (`"messenger"`, `"servant"`, `"clown"`, `"ghost"`, …). |
| `SKIP_FRAGMENTS` ([:37](build_character_aliases.py#L37)) | `<b>` content fragments that signal stage labels rather than speaker names (`"act "`, `"exit"`, `"enter "`, …). |
| `speakers_in_file(path)` ([:40](build_character_aliases.py#L40)) | Returns the set of distinct cleaned speaker strings in one HTML file. |
| `main()` ([:58](build_character_aliases.py#L58)) | Walks `books/`, skips the `Poetry/` subdirectory, builds the per-play speaker set, inverts to `name → set(work_codes)`, and keeps names where that set has size 1 (and length ≥ 4, not a stoplist hit, not pure digits). Writes the JSON and prints a per-play summary. |

### analyse_queries.py — metrics reporter

Reads `query_metrics.jsonl` and prints summary statistics.

| Symbol | Purpose |
|---|---|
| `percentile(xs, p)` ([:16](analyse_queries.py#L16)) | Linear-interpolated percentile (no numpy dependency). |
| `summary_line(label, vals)` ([:24](analyse_queries.py#L24)) | Renders `n=… mean=… p50=… p90=… p99=… max=…`. |
| `histogram(vals, buckets)` ([:38](analyse_queries.py#L38)) | ASCII histogram with given bucket upper edges. |
| `main()` ([:57](analyse_queries.py#L57)) | Loads rows (optionally only the last `--tail N`), prints overall stats, single-turn vs follow-up comparison, per-`work_code` breakdown, and a `total_s` histogram. |

CLI: `venv/bin/python analyse_queries.py [--tail N] [--metrics PATH]`.

## Ollama integration

Ollama runs as a separate daemon (typically `ollama serve`) on `localhost:11434`.
The app talks to it via the two LlamaIndex adapters:

| Use | Adapter | Model | Where configured |
|---|---|---|---|
| Embedding (ingestion + query-time) | `OllamaEmbedding(model_name=…)` | `mxbai-embed-large` | [Shakespeare.py:128](Shakespeare.py#L128), [BardWorksSetup.py:153](BardWorksSetup.py#L153) |
| Chat / condense LLM | `Ollama(model=…, request_timeout=120.0, keep_alive=KEEP_ALIVE)` | `gemma2:2b` | [Shakespeare.py:129-133](Shakespeare.py#L129-L133) |

Both must be **pulled** before the first run:

```
ollama pull mxbai-embed-large
ollama pull gemma2:2b
```

**Embedding model parity.** Whatever embedding model is configured at ingestion
time must also be configured at query time — vectors written by one model are
unusable by another. Both files reference the same `EMBED_MODEL` constant for
this reason. Changing it requires re-ingesting with `--reset`.

**Keep-alive.** `KEEP_ALIVE = "24h"` instructs Ollama to keep `gemma2:2b`
resident in memory between requests, eliminating cold-load latency on
subsequent queries. A `Settings.llm.complete("Hello")` warm-up runs at import
time so the very first user query also benefits.

**Two LLM calls per turn.** `CondensePlusContextChatEngine` issues **two** LLM
calls per follow-up turn: once to condense the question against prior history,
once to generate the answer. Single-turn questions usually skip or short-circuit
the condense call. `analyse_queries.py` separates `single-turn` from `follow-up`
totals for exactly this reason — follow-ups are typically ~2× the cost of
single-turns.

## ChromaDB layout

Chroma is opened as a `PersistentClient(path=CHROMA_DIR)`, which writes a
sqlite database plus blob files under `chroma_db/`. All documents live in one
collection named `shakespeare`.

**Metadata schema** (set by `BardWorksSetup.py`'s parsers):

| Field | Type | Source | Used by retriever? |
|---|---|---|---|
| `file_name` | str | `path.name` | shown in citations |
| `file_path` | str | absolute path | — |
| `play` | str | `<td class="play">` text or `<title>` for poetry | — |
| `work_code` | str | filename prefix (`macbeth`, `1henryiv`, …) | **yes — filter key** |
| `act` | int | filename component, `0` for poetry | — |
| `scene` | int | filename component, `0` for poetry | — |
| `work_type` | `"play"` \| `"poetry"` | poetry path heuristic | — |

The `work_code` field is the **join key** between ingestion-time metadata and
runtime routing: every `(alias → work_code)` entry in `PLAY_ALIASES` and every
`(name → work_code)` entry in `character_aliases.json` must match a value
written here.

## Configuration

All knobs are module-level constants — no env vars, no config file. Edit and
restart.

### Shakespeare.py

| Constant | Default | Notes |
|---|---|---|
| `COLLECTION_NAME` | `"shakespeare"` | Must match the value used at ingestion. |
| `EMBED_MODEL` | `"mxbai-embed-large"` | Must match ingestion. Changing requires re-ingest with `--reset`. |
| `LLM_MODEL` | `"gemma2:2b"` | Any Ollama chat model works. Larger = better answers, slower TTFT. |
| `TOP_K` | `3` | Passages fetched per query. Raising costs more LLM tokens but reduces "answer not in context" misses. |
| `KEEP_ALIVE` | `"24h"` | Ollama model-resident duration. |
| `METRICS_PATH` | `./query_metrics.jsonl` | Where per-query rows are appended. |
| `ChatMemoryBuffer(token_limit=3000)` ([:191](Shakespeare.py#L191)) | History budget — older turns are dropped when summed token count exceeds this. |

### BardWorksSetup.py

| Constant | Default | Notes |
|---|---|---|
| `COLLECTION_NAME` | `"shakespeare"` | |
| `EMBED_MODEL` | `"mxbai-embed-large"` | |
| `BOILERPLATE_FRAGMENTS` | nav/footer strings | Add new entries here if a new source format introduces other boilerplate. |
| `SCENE_FILENAME` | regex | Adjust if filenames diverge from `<work>.<act>.<scene>.html`. |

## Setup and installation

**Prerequisites**

- Python 3.10+ (uses `str | None` PEP 604 syntax).
- An NVIDIA GPU is optional but recommended — `gemma2:2b` runs comfortably on a GTX 1050 Ti.
- A running Ollama daemon: install from <https://ollama.com>, then `ollama serve`.

**Steps**

```sh
# 1. Clone & enter the project
git clone https://github.com/richardpaskin/shakespeare
cd shakespeare

# 2. Python venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Pull the Ollama models
ollama pull mxbai-embed-large
ollama pull gemma2:2b

# 4. Put source files in ./books/ (HTML layout: books/<work_code>/<work>.<act>.<scene>.html
#    plus books/Poetry/*.html). Not in the repo — gitignored.

# 5. Build the vector index (one-off)
python BardWorksSetup.py --filetype html --reset

# 6. (Optional) Rebuild character aliases — already committed, but regenerate
#    if you've changed the source corpus
python build_character_aliases.py

# 7. Launch the UI
python Shakespeare.py
# → http://127.0.0.1:7860
```

To monitor GPU usage while serving: `nvidia-smi -l 1` or
`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.

To inspect query timings: `python analyse_queries.py` (or `--tail 50`).

## Dependencies

Pinned by the package managers of each, not by version in `requirements.txt`:

| Package | Used for |
|---|---|
| `gradio` | UI / `ChatInterface`. |
| `chromadb` | Persistent vector store. |
| `llama-index` | RAG orchestration: `VectorStoreIndex`, `BaseRetriever`, `CondensePlusContextChatEngine`, `ChatMemoryBuffer`, `MetadataFilters`. |
| `llama-index-vector-stores-chroma` | Chroma adapter. |
| `llama-index-llms-ollama` | Ollama chat adapter. |
| `llama-index-embeddings-ollama` | Ollama embedding adapter. |
| `beautifulsoup4` | HTML parsing in `BardWorksSetup.py` and `build_character_aliases.py`. |

External services: an **Ollama daemon** is required at runtime — it's not a
Python dependency and isn't installed by `pip`.

## Files at runtime

| Path | Created by | Purpose |
|---|---|---|
| `chroma_db/` | `BardWorksSetup.py` | Persistent vector store. |
| `character_aliases.json` | `build_character_aliases.py` | Character-name routing table. Committed. |
| `query_metrics.jsonl` | `Shakespeare.py` `_record_metric` | One JSON row per query. Gitignored. |
| `llama_index.log` | `Shakespeare.py` `logging.basicConfig` | All `shakespeare` logger output + LlamaIndex internals. Gitignored. |
| `shakespeare_ui.log` | redirect when launching | Gradio stdout/stderr when started in the background. Gitignored. |

## Known design choices and trade-offs

- **Per-query retriever instance.** [`answer`](Shakespeare.py#L232) constructs a new `PlayFilterRetriever` and a new `CondensePlusContextChatEngine` on every request. This is intentional: the retriever stashes per-call routing state (`last_query`, `last_work_code`) that the metrics code reads after the stream completes. Concurrent requests would otherwise race on these fields.
- **Routing precedence: play > character.** An explicit play name is treated as a stronger signal than a character name. A query like *"Tybalt in Macbeth"* routes to Macbeth — this is the intended ranking. The character regex only runs if `PLAY_ALIASES` finds nothing.
- **Word-boundary matching for characters, substring for plays.** Character names are single tokens that may appear inside English words, so they need `\b…\b`. Multi-word play titles ("twelfth night", "much ado about nothing") are distinctive enough that bare substring matching is safe and simpler.
- **Common-name false positives.** Some character names overlap with everyday English (`"adam"`, `"adrian"`). Misrouting is rare in Shakespeare-flavoured questions but possible — extend the `STOPLIST` in [build_character_aliases.py](build_character_aliases.py#L28) if it bites.
- **Sources are stripped from memory.** `build_memory` removes the trailing `**References:**` block from prior assistant turns so the LLM doesn't learn to fabricate citation lines verbatim. The bare answer text is what re-enters the condense step's context.
