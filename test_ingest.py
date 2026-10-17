from pathlib import Path

import pytest

from assistant import build_prompt, retrieve
from ingest import Document, build_index, chunk_documents, chunk_text

SAMPLE = " ".join(f"word{i}" for i in range(400))  # ~2,700 characters of distinct words


def test_chunks_are_overlapping_and_the_expected_size() -> None:
    chunks = chunk_text(SAMPLE, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert 0 < len(chunk) <= 200
    for previous, current in zip(chunks, chunks[1:]):
        # The start of each chunk repeats the end of the one before it.
        first_word = current.split()[0]
        assert first_word in previous.split()
        overlap = previous[previous.index(first_word):]
        assert current.startswith(overlap)
        assert 30 <= len(overlap) <= 50  # about `overlap`, adjusted to word boundaries


def test_chunks_cover_all_the_text() -> None:
    chunks = chunk_text(SAMPLE, chunk_size=200, overlap=50)
    words_seen = {word for chunk in chunks for word in chunk.split()}
    assert words_seen == set(SAMPLE.split())


def test_chunks_never_split_words() -> None:
    for chunk in chunk_text(SAMPLE, chunk_size=137, overlap=31):
        assert all(word.startswith("word") and word[4:].isdigit() for word in chunk.split())


def test_short_and_empty_text() -> None:
    assert chunk_text("just a few words", chunk_size=200, overlap=50) == ["just a few words"]
    assert chunk_text("   \n  ", chunk_size=200, overlap=50) == []


@pytest.mark.parametrize("chunk_size, overlap", [(100, 100), (100, 150), (0, 0), (100, -1)])
def test_invalid_settings_are_rejected(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text(SAMPLE, chunk_size=chunk_size, overlap=overlap)


def test_chunk_ids_are_unique_across_documents() -> None:
    docs = [Document("a.md", SAMPLE), Document("b.md", SAMPLE)]
    ids, texts, metadatas = chunk_documents(docs, chunk_size=200, overlap=50)
    assert len(ids) == len(set(ids)) == len(texts) == len(metadatas)
    assert metadatas[0] == {"source": "a.md", "chunk": 0}


def test_build_prompt_labels_each_excerpt() -> None:
    from assistant import Chunk

    chunks = [Chunk("x#0", "Dogs on Fridays.", "house-rules.md", 0.2)]
    prompt = build_prompt("Can I bring my dog?", chunks)
    assert '<excerpt source="house-rules.md">' in prompt
    assert prompt.rstrip().endswith("Question: Can I bring my dog?")


def test_retrieval_finds_the_right_document(tmp_path: Path) -> None:
    # Uses the real handbook and Chroma's local embedding model (no API key needed).
    collection = build_index(db_path=tmp_path / "db")
    top = retrieve("Can I bring my dog to work?", top_k=1, collection=collection)
    assert top[0].source == "house-rules.md"
    top = retrieve("How much is the Dedicated Desk?", top_k=2, collection=collection)
    assert "membership.md" in {chunk.source for chunk in top}
