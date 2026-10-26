"""
RAG (Retrieval-Augmented Generation) with LangChain + Ollama + ChromaDB
=====================================================================

How it works:

  INGEST (python app.py ingest)          ASK (python app.py)
  -----------------------------          -------------------
  docs/*.pdf, *.txt, *.md                You type a question
        |                                      |
        v                                      v
  Split into small chunks                Embed the question
        |                                      |
        v                                      v
  Embed each chunk (Ollama)  -------->   ChromaDB finds the TOP_K
        |                    same model  most similar chunks
        v                                      |
  Save vectors in ChromaDB                     v
  (chroma_db/ folder)                    LLM answers using ONLY those
                                         chunks, and we show the sources

Settings live in .env.
"""

import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# =====================================================
# STEP 1: Load settings from .env
# =====================================================

load_dotenv()

BASE_FOLDER = Path(__file__).parent
DOCS_FOLDER = BASE_FOLDER / os.getenv("DOCS_FOLDER", "docs")
CHROMA_FOLDER = BASE_FOLDER / os.getenv("CHROMA_FOLDER", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_documents")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))


# =====================================================
# STEP 2: Models and vector store
# =====================================================

# Turns text into vectors (lists of numbers). Similar meaning -> similar vectors.
# The SAME model must be used for ingesting and for questions.
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

# The LLM that writes the final answer. temperature=0 = no made-up creativity.
llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)


def open_vector_store() -> Chroma:
    """Open the ChromaDB collection saved on disk (creates it if missing)."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_FOLDER),
    )


# =====================================================
# STEP 3: Ingest = load documents -> split -> embed -> save
#
# ingest() is the main function; it calls the helpers below it in order:
#   3.1 load_documents()   read the files
#   3.2 split_documents()  cut them into chunks
#   3.3 embed + save       done inside ingest() by ChromaDB
# =====================================================

def ingest():
    """Rebuild the vector store from scratch from everything in the docs folder."""
    # 3.1 Load
    print(f"Reading documents from {DOCS_FOLDER} ...")
    documents = load_documents()
    if not documents:
        print("No .pdf, .txt or .md files found. Add some to the docs folder and try again.")
        return

    # 3.2 Split
    chunks = split_documents(documents)
    print(f"Split {len(documents)} pages/files into {len(chunks)} chunks.")

    # 3.3 Embed + save. Start fresh so deleted or edited files don't leave
    # old chunks behind. add_documents() embeds every chunk and stores it.
    if CHROMA_FOLDER.exists():
        shutil.rmtree(CHROMA_FOLDER)

    print(f"Embedding with {EMBEDDING_MODEL} and saving to {CHROMA_FOLDER.name}/ ...")
    open_vector_store().add_documents(chunks)
    print("Done. Run 'python app.py' to ask questions.")


def load_documents() -> list:
    """3.1 Read every .pdf, .txt and .md file in the docs folder.

    Returns LangChain Documents. A PDF gives one Document per page, and each
    Document remembers its source file (and page) in .metadata.
    """
    documents = []
    for path in sorted(DOCS_FOLDER.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            # One Document per page; page is 0-based like most PDF tools.
            for page_number, page in enumerate(PdfReader(path).pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": str(path), "page": page_number},
                    ))
        elif suffix in {".txt", ".md"}:
            documents.append(Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": str(path)},
            ))
        else:
            continue
        print(f"  loaded {path.name}")
    return documents


def split_documents(documents: list) -> list:
    """3.2 Cut documents into overlapping chunks small enough to embed and to fit in a prompt."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(documents)


# =====================================================
# STEP 4: Ask = retrieve relevant chunks -> LLM answers from them
#
# ask() is the main function: 4.1 retrieve -> 4.2 augment -> 4.3 generate.
# describe_source() below it is a small helper for labelling chunks.
# =====================================================

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions using ONLY the context below. "
     "If the answer is not in the context, say \"I couldn't find that in the documents.\" "
     "Keep answers concise.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def ask(question: str, vector_store: Chroma) -> tuple[str, list]:
    """Answer one question. Returns (answer, the chunks it was based on)."""
    # 4.1 Retrieve: find the TOP_K chunks most similar to the question.
    chunks = vector_store.similarity_search(question, k=TOP_K)

    # 4.2 Augment: put those chunks into the prompt, each labelled with its source.
    context = "\n\n".join(f"[{describe_source(c)}]\n{c.page_content}" for c in chunks)

    # 4.3 Generate: the LLM answers from that context only.
    answer = llm.invoke(PROMPT.format_messages(context=context, question=question))
    return answer.content, chunks


def describe_source(doc) -> str:
    """Turn a chunk's metadata into a readable label like 'guide.pdf, page 3'."""
    name = Path(doc.metadata.get("source", "unknown")).name
    page = doc.metadata.get("page")
    return f"{name}, page {page + 1}" if page is not None else name


# =====================================================
# STEP 5: Command line
# =====================================================

def ready_vector_store():
    """Open the vector store, building it first if it is empty. None if no documents."""
    vector_store = open_vector_store()
    if vector_store._collection.count() == 0:
        print("The vector store is empty, ingesting documents first.\n")
        ingest()
        vector_store = open_vector_store()
        if vector_store._collection.count() == 0:
            return None
    return vector_store


def print_answer(question: str, vector_store: Chroma):
    """Ask one question and print the answer and its sources."""
    try:
        answer, chunks = ask(question, vector_store)
        print(f"\nAssistant: {answer}\n")
        # Show where the answer came from (unique sources, in order),
        # unless the documents didn't contain the answer.
        if "couldn't find that in the documents" not in answer:
            sources = list(dict.fromkeys(describe_source(c) for c in chunks))
            print("Sources:", "; ".join(sources), "\n")
    except Exception as exc:
        print(f"\nError: {exc}\n")


def chat():
    """Question-and-answer loop in the terminal."""
    vector_store = ready_vector_store()
    if vector_store is None:
        return

    print("=" * 60)
    print("RAG CHAT  (LangChain + Ollama + ChromaDB)")
    print(f"Chunks indexed: {vector_store._collection.count()}   |   "
          f"LLM: {CHAT_MODEL}   |   Embeddings: {EMBEDDING_MODEL}")
    print("=" * 60)
    print("Type a question, or 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue
        print_answer(question, vector_store)


if __name__ == "__main__":
    # python app.py ingest            -> rebuild the vector store
    # python app.py "your question"   -> answer one question and exit
    # python app.py                   -> chat
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest()
    elif len(sys.argv) > 1:
        store = ready_vector_store()
        if store is not None:
            question = " ".join(sys.argv[1:])
            print(f"You: {question}")
            print_answer(question, store)
    else:
        chat()
