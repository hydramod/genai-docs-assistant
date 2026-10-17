"""Step 2: the RAG assistant - retrieve the most relevant chunks, then answer from them only.

Run it with:  uv run python assistant.py "How much is Resident membership?"
With no question, it runs a few demo questions, including one the handbook can't answer.
"""

import logging
import sys
from dataclasses import dataclass, field

from chromadb.api.models.Collection import Collection
from langfuse import get_client, observe

from ingest import open_collection
from llm import MISSING_CREDENTIALS, call_claude, flush_traces, has_credentials, response_text

logger = logging.getLogger(__name__)

# How many chunks to hand the model. The handbook only has 15 chunks, so 3 is a
# focused starting point. The eval measures whether more helps.
TOP_K = 3

SYSTEM_PROMPT = """\
You are the help desk assistant for Fernhill Works, a coworking space. Members ask you \
questions and you answer using only the handbook excerpts provided with each question.

Rules:
- Base every statement on the excerpts. Don't use outside knowledge or guess at \
details such as prices, times or policies that the excerpts don't state.
- If the excerpts don't contain the answer, say plainly that the handbook doesn't \
cover it and suggest asking at reception. Don't make up an answer.
- If the excerpts answer only part of the question, answer that part and say which \
part isn't covered.
- Keep answers short and direct: one to three sentences unless the question needs more.
"""


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    distance: float  # cosine distance: 0 means identical meaning, higher is less similar


@dataclass
class Answer:
    text: str
    sources: list[str] = field(default_factory=list)


@observe(name="retrieve", as_type="retriever")
def retrieve(question: str, top_k: int = TOP_K, collection: Collection | None = None) -> list[Chunk]:
    """Return the top_k chunks whose meaning is closest to the question."""
    collection = collection or open_collection()
    results = collection.query(query_texts=[question], n_results=top_k)
    chunks = [
        Chunk(id=chunk_id, text=text, source=metadata["source"], distance=distance)
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
    logger.info(
        "Retrieved %d chunks: %s",
        len(chunks),
        ", ".join(f"{c.id} ({c.distance:.2f})" for c in chunks),
    )
    get_client().update_current_span(output=[c.__dict__ for c in chunks])
    return chunks


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Put the excerpts first and the question last, each excerpt labelled with its source."""
    excerpts = "\n\n".join(
        f'<excerpt source="{chunk.source}">\n{chunk.text}\n</excerpt>' for chunk in chunks
    )
    return f"Handbook excerpts:\n\n{excerpts}\n\nQuestion: {question}"


@observe(name="generate")
def generate(question: str, chunks: list[Chunk]) -> str:
    response = call_claude(
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
        system=SYSTEM_PROMPT,
    )
    return response_text(response)


@observe(name="answer_question")
def answer_question(question: str, top_k: int = TOP_K) -> Answer:
    """Answer a question about Fernhill Works using only the handbook."""
    logger.info("Question: %s", question)
    chunks = retrieve(question, top_k)
    text = generate(question, chunks)
    sources = sorted({chunk.source for chunk in chunks})
    return Answer(text=text, sources=sources)


DEMO_QUESTIONS = [
    "How much does Resident membership cost?",
    "Can I bring my dog in on a Wednesday?",
    "Which meeting room should I book for a team of 10?",
    "Is there a gym at Fernhill Works?",  # not in the handbook: it should say so
]


def main() -> int:
    if not has_credentials():
        logger.error(MISSING_CREDENTIALS)
        return 1
    questions = sys.argv[1:] or DEMO_QUESTIONS
    for question in questions:
        answer = answer_question(question)
        print(f"\nQ: {question}\nA: {answer.text}\n   (sources: {', '.join(answer.sources)})")
    flush_traces()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    sys.exit(main())
