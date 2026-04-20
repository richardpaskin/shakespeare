from pathlib import Path

import chromadb
import gradio as gr
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

PROJECT_DIR = Path(__file__).parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "shakespeare"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
TOP_K = 4


def load_query_engine():
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120.0)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index.as_query_engine(similarity_top_k=TOP_K)


query_engine = load_query_engine()


def format_sources(response) -> str:
    if not response.source_nodes:
        return ""
    lines = ["", "**References:**"]
    for node in response.source_nodes:
        meta = node.node.metadata
        file_name = meta.get("file_name", "unknown")
        page = meta.get("page_label", "?")
        snippet = node.node.get_content()[:200].replace("\n", " ").strip()
        score = f" (score {node.score:.2f})" if node.score is not None else ""
        lines.append(f"- **{file_name}**, p. {page}{score}: _{snippet}…_")
    return "\n".join(lines)


def answer(message, history):
    response = query_engine.query(message)
    return f"{response}\n{format_sources(response)}"


demo = gr.ChatInterface(
    fn=answer,
    title="Shakespeare Q&A",
    description="Ask questions about the complete works of Shakespeare. Answers cite their source PDFs.",
)


if __name__ == "__main__":
    demo.launch()
