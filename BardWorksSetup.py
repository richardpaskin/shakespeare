import argparse
from pathlib import Path

import chromadb
from bs4 import BeautifulSoup
from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

PROJECT_DIR = Path(__file__).parent
BOOKS_DIR = PROJECT_DIR / "books"
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "shakespeare"
EMBED_MODEL = "nomic-embed-text"
HTML_EXTS = (".html", ".htm")


def load_pdf_documents(source_dir: Path) -> list[Document]:
    return SimpleDirectoryReader(
        input_dir=str(source_dir),
        required_exts=[".pdf"],
        recursive=True,
    ).load_data()


def load_html_documents(source_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in HTML_EXTS):
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if not text:
            continue
        documents.append(
            Document(
                text=text,
                metadata={"file_name": path.name, "file_path": str(path)},
            )
        )
    return documents


LOADERS = {
    "pdf": load_pdf_documents,
    "html": load_html_documents,
}


def build_index(source_dir: Path, chroma_dir: Path, filetype: str) -> None:
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

    documents = LOADERS[filetype](source_dir)
    print(f"Loaded {len(documents)} {filetype} documents from {source_dir}")
    if not documents:
        print("No documents to index. Exiting.")
        return

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print(f"Index persisted to {chroma_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest Shakespeare source files into a persistent ChromaDB index."
    )
    parser.add_argument(
        "--filetype",
        choices=sorted(LOADERS.keys()),
        default="pdf",
        help="Which file type to ingest from the source directory.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=BOOKS_DIR,
        help="Directory to read source files from (default: ./books).",
    )
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    args = parser.parse_args()

    args.source_dir.mkdir(exist_ok=True)
    build_index(args.source_dir, args.chroma_dir, args.filetype)
