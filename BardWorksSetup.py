import argparse
import re
from pathlib import Path

import chromadb
from bs4 import BeautifulSoup, NavigableString
from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

PROJECT_DIR = Path(__file__).parent
BOOKS_DIR = PROJECT_DIR / "books"
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "shakespeare"
EMBED_MODEL = "mxbai-embed-large"
HTML_EXTS = (".html", ".htm")

SCENE_FILENAME = re.compile(
    r"^(?P<work>[^.]+)\.(?P<act>\d+)\.(?P<scene>\d+)\.html?$",
    re.IGNORECASE,
)
BOILERPLATE_FRAGMENTS = (
    "Shakespeare homepage",
    "Previous scene",
    "Next scene",
    "Previous section",
    "Next section",
)


def load_pdf_documents(source_dir: Path) -> list[Document]:
    return SimpleDirectoryReader(
        input_dir=str(source_dir),
        required_exts=[".pdf"],
        recursive=True,
    ).load_data()


def _clean_soup(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()
    # The nav row sits in a <table>; drop the whole table so we don't keep the "play" cell alone either
    for nav_td in soup.find_all("td", class_="nav"):
        table = nav_td.find_parent("table")
        if table:
            table.decompose()


def _scrub_text(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines()
        if line.strip() and not any(frag in line for frag in BOILERPLATE_FRAGMENTS)
    )


def _first_text_child(tag) -> str | None:
    if tag is None:
        return None
    for child in tag.children:
        if isinstance(child, NavigableString):
            s = str(child).strip()
            if s:
                return s
    return None


def _parse_play_html(path: Path) -> Document | None:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")

    # Extract the play name BEFORE we strip the nav. Malformed HTML (unclosed <td>)
    # means the play cell has the nav table nested inside it as a child, so
    # get_text() would pull in all the boilerplate. Read only the first direct
    # text child.
    play_name = _first_text_child(soup.find("td", class_="play")) or path.parent.name

    _clean_soup(soup)

    m = SCENE_FILENAME.match(path.name)
    if m:
        work_code = m.group("work")
        act = int(m.group("act"))
        scene = int(m.group("scene"))
    else:
        work_code = path.stem
        act = scene = 0

    text = _scrub_text(soup.get_text(separator="\n", strip=True))
    if not text:
        return None

    header = f"From {play_name}"
    if act or scene:
        header += f", Act {act}, Scene {scene}"
    header += ".\n\n"

    return Document(
        text=header + text,
        metadata={
            "file_name": path.name,
            "file_path": str(path),
            "play": play_name,
            "work_code": work_code,
            "act": act,
            "scene": scene,
            "work_type": "play",
        },
    )


def _parse_poetry_html(path: Path) -> Document | None:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else path.stem

    _clean_soup(soup)

    text = _scrub_text(soup.get_text(separator="\n", strip=True))
    if not text:
        return None

    header = f"From {title} (Shakespeare, poetry).\n\n"
    return Document(
        text=header + text,
        metadata={
            "file_name": path.name,
            "file_path": str(path),
            "play": title,
            "work_code": path.stem,
            "work_type": "poetry",
        },
    )


def load_html_documents(source_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in HTML_EXTS):
        is_poetry = any(part.lower() == "poetry" for part in path.parts)
        parser = _parse_poetry_html if is_poetry else _parse_play_html
        doc = parser(path)
        if doc:
            documents.append(doc)
    return documents


LOADERS = {
    "pdf": load_pdf_documents,
    "html": load_html_documents,
}


def build_index(source_dir: Path, chroma_dir: Path, filetype: str, reset: bool) -> None:
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

    documents = LOADERS[filetype](source_dir)
    print(f"Loaded {len(documents)} {filetype} documents from {source_dir}")
    if not documents:
        print("No documents to index. Exiting.")
        return

    client = chromadb.PersistentClient(path=str(chroma_dir))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except (ValueError, Exception):
            pass
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
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the existing collection before ingesting (prevents duplicate embeddings on re-run).",
    )
    args = parser.parse_args()

    args.source_dir.mkdir(exist_ok=True)
    build_index(args.source_dir, args.chroma_dir, args.filetype, args.reset)
