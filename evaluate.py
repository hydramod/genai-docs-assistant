"""Step 5: evaluate the assistant. Run every question in eval_set.json through it,
have Claude judge each answer from 1 to 5, and report the average.

Run it with:
    uv run python evaluate.py --top-k 2 --label baseline
    uv run python evaluate.py --top-k 5 --label more-chunks --baseline eval_results/baseline.json

Results are saved to eval_results/<label>.json so runs can be compared.
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from langfuse import get_client, observe

from agent import run_agent
from assistant import TOP_K, answer_question
from ingest import load_documents
from llm import MISSING_CREDENTIALS, MODEL, call_claude, flush_traces, has_credentials, response_text

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
EVAL_SET_PATH = PROJECT_DIR / "eval_set.json"
RESULTS_DIR = PROJECT_DIR / "eval_results"
LOW_SCORE = 3  # answers scoring this or below get printed with the judge's reason

JUDGE_INSTRUCTIONS = """\
You grade answers from a help desk assistant for a coworking space. The assistant \
must answer only from the space's handbook, and must say so when the handbook \
doesn't cover something.

You get the question, notes on what a good answer contains, and the assistant's \
answer. The full handbook is below as the source of truth. Score from 1 to 5:

5 - Correct and complete: covers everything the notes require, nothing wrong or invented.
4 - Correct, but misses a minor detail or adds unhelpful padding.
3 - Partly right: the main point is there but something required is missing or unclear.
2 - Mostly wrong or unhelpful, or includes a detail that isn't in the handbook.
1 - Wrong, or invents an answer when the handbook doesn't cover the question.

The notes say what the answer must contain. Use the handbook to check everything \
else: an extra detail that the handbook states is fine and must not lower the score. \
A detail the handbook doesn't state, or that contradicts it, counts as invented. \
Give your reason first, then the score.
"""


def judge_system_prompt() -> str:
    # The whole handbook is small (about 2,000 tokens), so the judge can see all of
    # it. Without it, the judge can only compare against the short notes, and it
    # wrongly marked true handbook facts (like the 10% annual discount) as invented.
    handbook = "\n\n".join(
        f'<document source="{doc.source}">\n{doc.text}\n</document>' for doc in load_documents()
    )
    return f"{JUDGE_INSTRUCTIONS}\n<handbook>\n{handbook}\n</handbook>"

# Structured output: the API guarantees the judge's reply matches this schema.
# The reason comes before the score so the judge explains before it decides.
JUDGE_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        },
        "required": ["reason", "score"],
        "additionalProperties": False,
    },
}


@dataclass
class EvalCase:
    question: str
    good_answer: str


@dataclass
class Judgement:
    score: int
    reason: str


@dataclass
class EvalResult:
    question: str
    good_answer: str
    answer: str
    score: int
    reason: str


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[EvalCase]:
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found: {path}")
    cases = [EvalCase(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
    logger.info("Loaded %d eval questions", len(cases))
    return cases


@observe(name="judge", as_type="evaluator")
def judge(question: str, good_answer: str, answer: str) -> Judgement:
    """Score one answer from 1 to 5 against the notes on what a good answer contains."""
    prompt = (
        f"<question>{question}</question>\n\n"
        f"<good_answer_notes>{good_answer}</good_answer_notes>\n\n"
        f"<assistant_answer>{answer}</assistant_answer>"
    )
    response = call_claude(
        messages=[{"role": "user", "content": prompt}],
        system=judge_system_prompt(),
        effort="medium",
        output_format=JUDGE_SCHEMA,
    )
    data = json.loads(response_text(response))
    return Judgement(score=data["score"], reason=data["reason"])


@observe(name="eval_case")
def evaluate_case(case: EvalCase, top_k: int, use_agent: bool) -> EvalResult:
    """Answer one eval question and judge it, all inside one Langfuse trace."""
    if use_agent:
        answer, _ = run_agent(case.question)
    else:
        answer = answer_question(case.question, top_k=top_k).text
    judgement = judge(case.question, case.good_answer, answer)
    # Attach the score to this question's trace so it shows up in Langfuse too.
    get_client().score_current_trace(name="judge_score", value=judgement.score)
    return EvalResult(**asdict(case), answer=answer, **asdict(judgement))


def run_eval(cases: list[EvalCase], top_k: int, use_agent: bool) -> list[EvalResult]:
    results = []
    for i, case in enumerate(cases, start=1):
        result = evaluate_case(case, top_k, use_agent)
        logger.info("[%d/%d] score %d: %s", i, len(cases), result.score, case.question)
        results.append(result)
    return results


def average_score(results: list[EvalResult]) -> float:
    return sum(r.score for r in results) / len(results)


def report(results: list[EvalResult], baseline: list[dict] | None = None) -> None:
    print(f"\nAverage score: {average_score(results):.2f} / 5 over {len(results)} questions")

    low = [r for r in results if r.score <= LOW_SCORE]
    if low:
        print(f"\n{len(low)} low-scoring answers (score <= {LOW_SCORE}):")
        for r in low:
            print(f"\n  [{r.score}] {r.question}\n  Answer: {r.answer}\n  Judge:  {r.reason}")
    else:
        print(f"No answers scored {LOW_SCORE} or below.")

    if baseline:
        before = {item["question"]: item["score"] for item in baseline}
        baseline_avg = sum(before.values()) / len(before)
        change = average_score(results) - baseline_avg
        print(f"\nCompared with baseline: {baseline_avg:.2f} -> {average_score(results):.2f} ({change:+.2f})")
        for r in results:
            if r.question in before and before[r.question] != r.score:
                print(f"  {before[r.question]} -> {r.score}  {r.question}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the docs assistant")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="chunks to retrieve per question")
    parser.add_argument("--agent", action="store_true", help="evaluate the LangGraph agent instead of plain RAG")
    parser.add_argument("--label", default="latest", help="name for the saved results file")
    parser.add_argument("--baseline", type=Path, help="earlier results file to compare against")
    args = parser.parse_args()

    if not has_credentials():
        logger.error(MISSING_CREDENTIALS)
        return 1
    try:
        cases = load_eval_set()
        baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1

    results = run_eval(cases, top_k=args.top_k, use_agent=args.agent)
    report(results, baseline)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{args.label}.json"
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    logger.info(
        "Saved results to %s (model=%s, top_k=%d, agent=%s)",
        out_path.relative_to(PROJECT_DIR),
        MODEL,
        args.top_k,
        args.agent,
    )
    flush_traces()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    sys.exit(main())
