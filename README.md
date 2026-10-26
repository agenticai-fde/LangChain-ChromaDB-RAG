# RAG with LangChain + Ollama + ChromaDB

Ask questions about your own documents (PDF, TXT, Markdown). The app finds the
most relevant passages and a local LLM answers **only from those passages**,
showing which file and page each answer came from. Everything runs on your
computer: no API keys, no cloud, no cost.

| Part | Tool |
|---|---|
| Framework | LangChain |
| Chat model (writes answers) | Ollama `llama3.2` |
| Embedding model (text → vectors) | Ollama `nomic-embed-text` |
| Vector database | ChromaDB, saved to disk in `chroma_db/` |
| PDF reading | `pypdf` |

---

## What is RAG?

An LLM only knows what it was trained on. It has never seen *your* documents.
**Retrieval-Augmented Generation** fixes that in three moves:

1. **Retrieve:** find the passages in your documents that relate to the question.
2. **Augment:** paste those passages into the prompt.
3. **Generate:** the LLM answers using that text, and is told to say so if the answer isn't there.

To find "related" passages fast, every passage is turned into a vector (a list
of numbers) by an **embedding model**. Passages with similar meaning get
similar vectors, and a **vector database** (ChromaDB) can find the closest ones
in milliseconds.

---

## Architecture

The app has two phases. **Ingest** runs once (and again when documents change).
**Ask** runs for every question.

```
 PHASE 1: INGEST   python app.py ingest           PHASE 2: ASK   python app.py "question"
 ==================================               =========================================

  docs/                                             "How do I resume a conversation?"
  ├── guide.pdf                                                   |
  ├── notes.txt                                                   v
  └── faq.md                                         +--------------------------+
        |                                            | Embed the question       |
        v                                            | nomic-embed-text         |
  +---------------------------+                      +--------------------------+
  | 3.1 Load files            |                                   |
  |     pypdf (1 doc / page)  |                                   v
  |     .txt / .md as text    |                      +--------------------------+
  +---------------------------+                      | 4.1 RETRIEVE             |
        |                                            | ChromaDB similarity      |
        v                                            | search -> TOP_K (4)      |
  +---------------------------+                      | closest chunks           |
  | 3.2 Split into chunks     |                      +--------------------------+
  |     1000 chars each,      |                                   |
  |     150 chars overlap     |                                   v
  +---------------------------+                      +--------------------------+
        |                                            | 4.2 AUGMENT              |
        v                                            | prompt = rules           |
  +---------------------------+                      |        + chunks (with    |
  | 3.3 Embed each chunk      |                      |          file/page)      |
  |     nomic-embed-text      |                      |        + question        |
  +---------------------------+                      +--------------------------+
        |                                                         |
        v                                                         v
  +---------------------------+   searched at        +--------------------------+
  | ChromaDB                  |   question time      | 4.3 GENERATE             |
  | chroma_db/ on disk        | -------------------> | llama3.2 answers ONLY    |
  | (text + vector + source)  |                      | from the chunks          |
  +---------------------------+                      +--------------------------+
                                                                  |
                                                                  v
                                                   Answer + Sources (file, page)
                                                   or "I couldn't find that in
                                                       the documents."
```

**Why the same embedding model on both sides?** ChromaDB compares the
question's vector with the chunk vectors. Vectors from two different models
are not comparable, so if you change `EMBEDDING_MODEL` you must re-run ingest.

**Why overlap the chunks?** A sentence cut in half at a chunk border would lose
its meaning. The 150-character overlap means each border sentence appears whole
in at least one chunk.

---

## How the code is organised

Everything is in `app.py`, in the same order as the diagram. Each main function
comes first, followed by the helpers it calls.

```
app.py
│
├── STEP 1  Settings            load .env
├── STEP 2  Models + store      embeddings, llm, open_vector_store()
│
├── STEP 3  Ingest
│   ├── ingest()                main function: 3.1 -> 3.2 -> 3.3
│   ├── load_documents()        3.1 read files
│   └── split_documents()       3.2 cut into chunks
│
├── STEP 4  Ask
│   ├── PROMPT                  the rules given to the LLM
│   ├── ask()                   main function: 4.1 retrieve -> 4.2 augment -> 4.3 generate
│   └── describe_source()       "guide.pdf, page 3" labels
│
└── STEP 5  Command line
    ├── ready_vector_store()    open the store, ingest first if empty
    ├── print_answer()          ask + print answer and sources
    ├── chat()                  interactive loop
    └── if __name__ == ...      picks: ingest / one question / chat
```

| Step | Function | What it does |
|---|---|---|
| 1 | — | Loads settings from `.env` |
| 2 | `open_vector_store()` | Opens the ChromaDB collection on disk, with the embedding model |
| 3 | `ingest()` | Rebuilds the vector store: load → split → embed and save |
| 3.1 | `load_documents()` | Reads every PDF (one document per page), TXT and MD file in `docs/` |
| 3.2 | `split_documents()` | Cuts documents into overlapping chunks |
| 4 | `ask()` | Retrieves the top chunks, builds the prompt, asks the LLM |
| | `describe_source()` | Turns chunk metadata into "file.pdf, page 3" |
| 5 | `ready_vector_store()` | Opens the store; ingests automatically if it is empty |
| | `print_answer()` | Prints the answer, plus sources unless nothing was found |
| | `chat()` | Interactive question loop |

---

## Project structure

```
rag_chromadb/
├── app.py            all the code
├── docs/             your documents (.pdf, .txt, .md); 2 sample PDFs included
├── chroma_db/        created by ingest: the saved vectors (safe to delete)
├── .env              settings
├── .env.example      template for .env
├── requirements.txt  Python packages
├── README.md
└── .gitignore
```

---

## Step-by-step setup

Open a terminal (PowerShell on Windows; in VS Code press **Ctrl+`**).

### Step 1: Check the prerequisites

```powershell
python --version      # needs 3.10 or newer
ollama --version      # needs Ollama installed: https://ollama.com/
```

### Step 2: Download the two models

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
ollama list           # both should be listed
```

Ollama usually runs in the background after install. If a later step says
`Connection refused`, run `ollama serve` in a separate terminal.

### Step 3: Go to the project folder

```powershell
cd path\to\rag_chromadb
```

### Step 4: Create a virtual environment and install packages

Windows:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

> On macOS / Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python`
> in every command below.

### Step 5: Create `.env`

```powershell
copy .env.example .env
```

(macOS / Linux: `cp .env.example .env`.) The defaults work as they are.
See [Settings](#settings) to change them.

### Step 6: Add your documents

Copy `.pdf`, `.txt` or `.md` files into `docs/` (sub-folders are included).
Two sample PDFs are already there:

- `LangGraph_Supabase_Ollama_Memory_Guide.pdf`: persistent chat memory with LangGraph + Supabase
- `supabase_multiagent_guide.pdf`: a supervisor + CRUD multi-agent system

> Scanned PDFs (photos of pages) have no extractable text and are skipped.
> Use PDFs where you can select the text.

### Step 7: Build the vector store (ingest)

```powershell
.venv\Scripts\python.exe app.py ingest
```

Expected output:

```
Reading documents from ...\rag_chromadb\docs ...
  loaded LangGraph_Supabase_Ollama_Memory_Guide.pdf
  loaded supabase_multiagent_guide.pdf
Split 21 pages/files into 39 chunks.
Embedding with nomic-embed-text and saving to chroma_db/ ...
Done. Run 'python app.py' to ask questions.
```

Run this again whenever you add, change or remove documents. It rebuilds
`chroma_db/` from scratch.

### Step 8: Ask questions

There are three ways to run the app:

| Command | What it does |
|---|---|
| `.venv\Scripts\python.exe app.py "your question"` | Answers **one** question and exits |
| `.venv\Scripts\python.exe app.py` | Interactive chat: type questions, `exit` to quit |
| `.venv\Scripts\python.exe app.py ingest` | Rebuilds the vector store |

Both question modes ingest automatically if the vector store is empty.

Example:

```powershell
.venv\Scripts\python.exe app.py "What is a LangGraph checkpointer used for?"
```

```
You: What is a LangGraph checkpointer used for?

Assistant: A LangGraph checkpointer is used to persist the conversation state,
saving the conversation history in a database (in this case, Supabase
PostgreSQL) so that it can be resumed when the application is restarted,
using the same conversation ID.

Sources: LangGraph_Supabase_Ollama_Memory_Guide.pdf, page 2; ..., page 3; ..., page 5; ..., page 1
```

Each answer takes roughly a minute on a laptop CPU, and seconds on a GPU.

---

## Prompts to try

Paste these into the terminal from inside the `rag_chromadb` folder.

### Memory guide (LangGraph + Supabase + Ollama)

```powershell
.venv\Scripts\python.exe app.py "What is a LangGraph checkpointer used for?"
.venv\Scripts\python.exe app.py "How do I resume a conversation after restarting the app?"
.venv\Scripts\python.exe app.py "What should the .env file contain for the memory project?"
.venv\Scripts\python.exe app.py "What is the difference between the checkpointer and a long-term memory store?"
.venv\Scripts\python.exe app.py "What tables does the checkpointer store in Supabase?"
```

| Question | A good answer mentions |
|---|---|
| checkpointer used for | saving conversation state in PostgreSQL / Supabase |
| resume a conversation | entering the **same conversation (thread) ID** |
| .env contents | `SUPABASE_DB_URI` and `OLLAMA_MODEL` |
| checkpointer vs long-term memory | checkpointer = one conversation; memory store = facts shared across conversations |
| tables in Supabase | checkpoint tables managed by LangGraph (thread ID, messages, metadata) |

### Multi-agent guide

```powershell
.venv\Scripts\python.exe app.py "Which Python packages does the multi-agent guide install?"
.venv\Scripts\python.exe app.py "What does the supervisor agent do?"
.venv\Scripts\python.exe app.py "List the five agents and what each one does"
.venv\Scripts\python.exe app.py "Why must the delete agent not trust an LLM-generated approval?"
.venv\Scripts\python.exe app.py "Why does the guide rename create_agent to make_agent?"
```

| Question | A good answer mentions |
|---|---|
| packages | langgraph, langchain, langchain-ollama, supabase, python-dotenv |
| supervisor | chooses which agent handles the request (routing) |
| five agents | create, read, update, delete, skill |
| delete approval | the **application** must enforce approval, not the LLM |
| make_agent | the variable `create_agent` would shadow the imported function |

### Across both documents

```powershell
.venv\Scripts\python.exe app.py "Which Ollama models do the two guides use?"
```

A good answer names `llama3.2` (memory guide) **and** `llama3.1` (multi-agent
guide), and the `Sources:` line lists **both** PDFs.

### Questions it should refuse (tests for hallucination)

```powershell
.venv\Scripts\python.exe app.py "What is the capital of France?"
.venv\Scripts\python.exe app.py "How do I deploy this to AWS Lambda?"
```

Correct behaviour: **"I couldn't find that in the documents."** and **no**
`Sources:` line. If it answers "Paris", the LLM is using outside knowledge
instead of the documents.

### Test with your own document

```powershell
"Our office opens at 9 AM and closes at 6 PM. The cafeteria is on floor 3. Parking is free for visitors on weekends." | Out-File docs\office.txt -Encoding utf8
.venv\Scripts\python.exe app.py ingest
.venv\Scripts\python.exe app.py "What time does the office open?"
.venv\Scripts\python.exe app.py "Where is the cafeteria?"
.venv\Scripts\python.exe app.py "Is parking free on weekdays?"
```

Expected: `9 AM` and `floor 3`, each with `Sources: office.txt`. For the
weekday parking question, a careful answer says the document only mentions
weekends. Delete `docs\office.txt` and re-run ingest to remove it.

---

## Classroom experiments

Change one setting in `.env`, re-run ingest if the table says so, and ask the
same question again to see the effect.

| Experiment | Change | Re-ingest? | What to observe |
|---|---|---|---|
| Less context | `TOP_K=1` | no | Answers get shorter or miss details; fewer sources |
| More context | `TOP_K=8` | no | More complete answers, slower, sometimes off-topic |
| Small chunks | `CHUNK_SIZE=300`, `CHUNK_OVERLAP=50` | **yes** | Many more chunks; precise but fragmented answers |
| Big chunks | `CHUNK_SIZE=3000` | **yes** | Few chunks; broad context, less precise retrieval |
| Bigger LLM | `CHAT_MODEL=qwen2.5:7b` (`ollama pull qwen2.5:7b`) | no | Better wording, much slower on CPU |

---

## Settings

All in `.env`:

| Setting | Default | Meaning |
|---|---|---|
| `DOCS_FOLDER` | `docs` | Folder to read documents from |
| `CHROMA_FOLDER` | `chroma_db` | Where ChromaDB stores vectors |
| `COLLECTION_NAME` | `my_documents` | ChromaDB collection name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model (**re-ingest after changing**) |
| `CHAT_MODEL` | `llama3.2` | Model that writes answers |
| `CHUNK_SIZE` | `1000` | Characters per chunk (**re-ingest after changing**) |
| `CHUNK_OVERLAP` | `150` | Characters shared between neighbouring chunks (**re-ingest after changing**) |
| `TOP_K` | `4` | Chunks retrieved per question |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` on port 11434 | Ollama isn't running. Start it with `ollama serve`. |
| `model "nomic-embed-text" not found` | Run `ollama pull nomic-embed-text` (same for `llama3.2`). |
| `No .pdf, .txt or .md files found` | Put files in `docs/` (or the folder set in `DOCS_FOLDER`). |
| A PDF loads but answers never use it | It may be a scanned PDF with no text layer. |
| "Couldn't find" for things that are in the docs | Re-run ingest after adding files; try `TOP_K=6`. |
| Errors after changing `EMBEDDING_MODEL` | Re-run ingest; the old vectors came from a different model. |
| `.venv\Scripts\Activate.ps1 cannot be loaded` | You don't need to activate: use `.venv\Scripts\python.exe` directly, as in this README. |
| Slow answers | Normal on CPU. Keep `CHAT_MODEL=llama3.2` and close other apps. |
