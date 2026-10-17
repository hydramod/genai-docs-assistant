"""Step 1: load the handbook, split it into overlapping chunks, and store them in Chroma.

Run it with:  uv run python ingest.py
Re-running rebuilds the collection from scratch, so it's safe to run after editing docs.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
DOCS_DIR = PROJECT_DIR / "docs"
DB_PATH = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "handbook"

# About 2-4 sentences per chunk: small enough that each one is about one thing,
# and big enough to carry a complete fact like a price and what it includes.
CHUNK_SIZE = 500
# The overlap repeats the end of each chunk at the start of the next one, so a fact
# that lands on a boundary still appears whole in at least one chunk.
CHUNK_OVERLAP = 100


@dataclass
class Document:
    source: str  # file name, e.g. "membership.md"
    text: str


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """Read every .md file in the docs folder."""
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs folder not found: {docs_dir}")
    documents = [
        Document(source=path.name, text=path.read_text(encoding="utf-8"))
        for path in sorted(docs_dir.glob("*.md"))
    ]
    if not documents:
        raise FileNotFoundError(f"No .md files found in {docs_dir}")
    logger.info("Loaded %d documents from %s", len(documents), docs_dir.name)
    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of at most chunk_size characters that overlap by about `overlap`.

    Chunks break at spaces, so no word is cut in half. That's why the overlap is
    "about" rather than exact: it shifts to the nearest word boundary.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size")

    text = " ".join(text.split())  # collapse newlines and repeated spaces
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            last_space = text.rfind(" ", start, end + 1)
            if last_space > start:
                end = last_space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break

        # Step back by the overlap, then forward to the start of the next whole word.
        next_start = end - overlap
        word_start = text.find(" ", next_start, end)
        if word_start != -1:
            next_start = word_start + 1
        # Always move forward, even if a single "word" is longer than the overlap.
        start = max(next_start, start + 1)
    return chunks


def chunk_documents(
    documents: list[Document], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> tuple[list[str], list[str], list[dict[str, str | int]]]:
    """Chunk every document. Returns parallel lists of ids, texts and metadata."""
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    for document in documents:
        for i, chunk in enumerate(chunk_text(document.text, chunk_size, overlap)):
            # The id says where the chunk came from, which helps when debugging retrieval.
            ids.append(f"{document.source}#{i}")
            texts.append(chunk)
            metadatas.append({"source": document.source, "chunk": i})
    logger.info("Split %d documents into %d chunks", len(documents), len(texts))
    return ids, texts, metadatas


def open_collection(db_path: Path = DB_PATH) -> Collection:
    """Open the Chroma collection stored on disk.

    Chroma's default embedding function runs a small model (all-MiniLM-L6-v2)
    locally, so embeddings need no API key. The first run downloads it (~80 MB).
    Cosine distance is the usual choice for comparing text embeddings.
    """
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def build_index(
    db_path: Path = DB_PATH,
    docs_dir: Path = DOCS_DIR,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Collection:
    """Rebuild the collection from the docs folder."""
    ids, texts, metadatas = chunk_documents(load_documents(docs_dir), chunk_size, overlap)

    client = chromadb.PersistentClient(path=str(db_path))
    # Start from empty each time so chunks from an older version of the docs (or an
    # older chunk size) can't linger alongside the new ones.
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    logger.info("Stored %d chunks in Chroma at %s", collection.count(), db_path.name)
    return collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Chroma index from docs/")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    collection = build_index(chunk_size=args.chunk_size, overlap=args.overlap)

    # Step 1.4: check a test query brings back sensible chunks.
    query = "Can I bring my dog to work?"
    results = collection.query(query_texts=[query], n_results=3)
    print(f"\nTest query: {query!r}")
    for chunk_id, text, distance in zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        print(f"  [{chunk_id}] distance={distance:.3f}  {text[:90]}...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    main()
