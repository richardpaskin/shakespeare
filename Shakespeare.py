import logging
from pathlib import Path

import chromadb
import gradio as gr
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

PROJECT_DIR = Path(__file__).parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"

logging.basicConfig(
    filename=PROJECT_DIR / "llama_index.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for noisy in ("httpx", "httpcore", "urllib3", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("shakespeare")
COLLECTION_NAME = "shakespeare"
EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "gemma2:2b"
TOP_K = 3
KEEP_ALIVE = "24h"

# Maps (lowercase alias) → work_code stored in metadata. Longest aliases first so
# "henry iv part 1" matches before "henry iv".
PLAY_ALIASES: list[tuple[str, str]] = sorted(
    [
        ("king lear", "lear"), ("lear", "lear"),
        ("hamlet", "hamlet"),
        ("macbeth", "macbeth"),
        ("othello", "othello"),
        ("midsummer night's dream", "midsummer"), ("midsummer", "midsummer"),
        ("romeo and juliet", "romeo_juliet"), ("romeo & juliet", "romeo_juliet"),
        ("julius caesar", "julius_caesar"),
        ("titus andronicus", "titus"), ("titus", "titus"),
        ("the tempest", "tempest"), ("tempest", "tempest"),
        ("merchant of venice", "merchant"),
        ("taming of the shrew", "taming_shrew"),
        ("twelfth night", "twelfth_night"),
        ("much ado about nothing", "much_ado"), ("much ado", "much_ado"),
        ("as you like it", "asyoulikeit"),
        ("all's well that ends well", "allswell"), ("alls well", "allswell"),
        ("measure for measure", "measure"),
        ("comedy of errors", "comedy_errors"),
        ("merry wives of windsor", "merry_wives"), ("merry wives", "merry_wives"),
        ("two gentlemen of verona", "two_gentlemen"),
        ("love's labour's lost", "lll"), ("loves labours lost", "lll"),
        ("winter's tale", "winters_tale"), ("winters tale", "winters_tale"),
        ("cymbeline", "cymbeline"),
        ("pericles", "pericles"),
        ("troilus and cressida", "troilus_cressida"),
        ("antony and cleopatra", "cleopatra"), ("antony & cleopatra", "cleopatra"),
        ("coriolanus", "coriolanus"),
        ("timon of athens", "timon"), ("timon", "timon"),
        ("henry iv part 1", "1henryiv"), ("1 henry iv", "1henryiv"),
        ("henry iv part 2", "2henryiv"), ("2 henry iv", "2henryiv"),
        ("henry v", "henryv"),
        ("henry vi part 1", "1henryvi"), ("1 henry vi", "1henryvi"),
        ("henry vi part 2", "2henryvi"), ("2 henry vi", "2henryvi"),
        ("henry vi part 3", "3henryvi"), ("3 henry vi", "3henryvi"),
        ("henry viii", "henryviii"),
        ("richard ii", "richardii"),
        ("richard iii", "richardiii"),
        ("king john", "john"),
    ],
    key=lambda p: -len(p[0]),
)


def detect_work_code(text: str) -> str | None:
    lower = text.lower()
    for alias, code in PLAY_ALIASES:
        if alias in lower:
            return code
    return None


def load_index():
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(
        model=LLM_MODEL,
        request_timeout=120.0,
        keep_alive=KEEP_ALIVE,
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store)


index = load_index()

# Warm the LLM so the first real query doesn't pay the cold-load cost.
Settings.llm.complete("Hello")


def _extract_text(obj) -> str:
    """Gradio may pass message content as a string, list (multimodal parts), or dict.
    Flatten any shape into a plain string."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        # Multimodal part like {"type": "text", "text": "..."}
        return _extract_text(obj.get("text") or obj.get("content"))
    if isinstance(obj, (list, tuple)):
        return " ".join(_extract_text(x) for x in obj if x)
    return str(obj)


def build_memory(history):
    """Replay Gradio's chat history into a LlamaIndex ChatMemoryBuffer."""
    mem = ChatMemoryBuffer.from_defaults(token_limit=3000)
    role_map = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT}
    for turn in history or []:
        if isinstance(turn, dict):
            role = role_map.get(turn.get("role"))
            content = _extract_text(turn.get("content"))
            if role and content:
                mem.put(ChatMessage(role=role, content=content))
        else:
            user_msg = _extract_text(turn[0]) if len(turn) > 0 else ""
            asst_msg = _extract_text(turn[1]) if len(turn) > 1 else ""
            if user_msg:
                mem.put(ChatMessage(role=MessageRole.USER, content=user_msg))
            if asst_msg:
                # Strip the References block we append so the LLM doesn't learn to repeat it
                clean = asst_msg.split("\n**References:**")[0]
                mem.put(ChatMessage(role=MessageRole.ASSISTANT, content=clean))
    return mem


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
    # Detect a play in the current message OR anywhere in prior history,
    # so follow-ups like "tell me about Edmund" still filter to King Lear.
    history_text = _extract_text(history)
    work_code = detect_work_code(message) or detect_work_code(history_text)
    log.info(
        "Q (history=%d turns, work_code=%s): %s",
        len(history or []),
        work_code,
        message,
    )

    filters = None
    if work_code:
        filters = MetadataFilters(
            filters=[MetadataFilter(key="work_code", value=work_code, operator=FilterOperator.EQ)]
        )
    retriever = index.as_retriever(similarity_top_k=TOP_K, filters=filters)
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        memory=build_memory(history),
    )
    response = chat_engine.stream_chat(message)
    for i, node in enumerate(response.source_nodes):
        meta = node.node.metadata
        snippet = node.node.get_content()[:300].replace("\n", " ").strip()
        log.info(
            "  retrieved[%d] score=%.3f file=%s page=%s: %s",
            i,
            node.score or 0.0,
            meta.get("file_name", "?"),
            meta.get("page_label", "?"),
            snippet,
        )
    text = ""
    for token in response.response_gen:
        text += token
        yield text
    log.info("A (%d chars): %s", len(text), text[:500].replace("\n", " "))
    yield f"{text}\n{format_sources(response)}"


CSS = """
textarea, input[type="text"] {
    border: 1px solid #bbb !important;
    border-radius: 6px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #4a90e2 !important;
    outline: none !important;
}
"""

with gr.Blocks(css=CSS) as demo:
    gr.ChatInterface(
        fn=answer,
        title="Shakespeare Q&A",
        description="Ask questions about the complete works of Shakespeare. Answers cite their source PDFs.",
    )


if __name__ == "__main__":
    demo.launch()
