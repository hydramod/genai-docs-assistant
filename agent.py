"""Step 3: a LangGraph agent that decides for itself when to search the handbook
and when to use the calculator.

Run it with:  uv run python agent.py "What would 3 extra hours in Cedar cost?"

The graph has two nodes in a loop:

    START -> model --(Claude asked for a tool)--> tools -> model -> ...
               \\--(Claude gave its final answer)--> END
"""

import ast
import json
import logging
import operator
import sys
from typing import Any, TypedDict

from langfuse import observe
from langgraph.graph import END, START, StateGraph

from assistant import TOP_K, retrieve
from llm import MISSING_CREDENTIALS, call_claude, flush_traces, has_credentials

logger = logging.getLogger(__name__)

# A safety limit: stop after this many model calls even if Claude keeps asking for tools.
MAX_MODEL_CALLS = 6

SYSTEM_PROMPT = """\
You are the help desk assistant for Fernhill Works, a coworking space.

You have two tools:
- search_handbook: searches the Fernhill Works member handbook. Use it for any \
question about the space: prices, memberships, rooms, hours, facilities or rules. \
Never answer those from memory - always search first, and you may search more than once.
- calculator: evaluates an arithmetic expression. Use it whenever an answer needs \
a calculation, rather than working it out in your head.

Answer only from what the handbook search returns. If the handbook doesn't cover \
something, say so plainly and suggest asking at reception. Keep answers short.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_handbook",
        "description": (
            "Search the Fernhill Works member handbook and return the most relevant "
            "excerpts. Covers membership prices, meeting rooms, opening hours, "
            "facilities and house rules."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up, in plain words."}
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate an arithmetic expression using + - * / // % ** and brackets, "
            "e.g. '3 * 35' or '(220 * 12) * 0.9'. Returns the numeric result."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
]


# --- Tools -----------------------------------------------------------------

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculate(expression: str) -> float:
    """Safely evaluate plain arithmetic.

    Never use eval() on text that came from a model: it would run any Python at all.
    Instead, parse the expression and allow only numbers and arithmetic operators.
    """

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")

    return evaluate(ast.parse(expression, mode="eval").body)


@observe(name="search_handbook", as_type="tool")
def search_handbook(query: str) -> str:
    chunks = retrieve(query, TOP_K)
    return "\n\n".join(f'<excerpt source="{c.source}">\n{c.text}\n</excerpt>' for c in chunks)


@observe(name="calculator", as_type="tool")
def calculator(expression: str) -> str:
    result = calculate(expression)
    # Show whole numbers without a trailing .0, and round away float noise.
    return str(int(result)) if float(result).is_integer() else str(round(result, 6))


TOOL_FUNCTIONS = {"search_handbook": search_handbook, "calculator": calculator}


def run_tool(name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    """Run one tool call. Returns (result text, is_error)."""
    try:
        return TOOL_FUNCTIONS[name](**tool_input), False
    except Exception as error:  # report the problem to Claude instead of crashing
        logger.warning("Tool %s failed: %s", name, error)
        return f"Error: {error}", True


# --- Graph -----------------------------------------------------------------


class AgentState(TypedDict):
    messages: list[dict[str, Any]]
    stop_reason: str
    model_calls: int
    tools_used: list[str]


def model_node(state: AgentState) -> dict[str, Any]:
    response = call_claude(messages=state["messages"], system=SYSTEM_PROMPT, tools=TOOLS)
    # Keep the full content (thinking and tool_use blocks too), not just the text:
    # the next request must replay the conversation exactly as Claude produced it.
    messages = state["messages"] + [{"role": "assistant", "content": response.content}]
    return {
        "messages": messages,
        "stop_reason": response.stop_reason,
        "model_calls": state["model_calls"] + 1,
    }


def tools_node(state: AgentState) -> dict[str, Any]:
    last_content = state["messages"][-1]["content"]
    results = []
    used = list(state["tools_used"])
    for block in last_content:
        if block.type != "tool_use":
            continue
        logger.info("Agent calls %s(%s)", block.name, json.dumps(block.input))
        text, is_error = run_tool(block.name, block.input)
        used.append(block.name)
        results.append(
            {"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": is_error}
        )
    # All results go back together in one user message.
    messages = state["messages"] + [{"role": "user", "content": results}]
    return {"messages": messages, "tools_used": used}


def next_step(state: AgentState) -> str:
    if state["stop_reason"] == "tool_use" and state["model_calls"] < MAX_MODEL_CALLS:
        return "tools"
    if state["stop_reason"] == "tool_use":
        logger.warning("Stopping after %d model calls without a final answer", MAX_MODEL_CALLS)
    return END


def build_graph() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", next_step, ["tools", END])
    graph.add_edge("tools", "model")
    return graph.compile()


AGENT = build_graph()


@observe(name="run_agent", as_type="agent")
def run_agent(question: str) -> tuple[str, list[str]]:
    """Answer a question, letting Claude choose which tools to use. Returns (answer, tools used)."""
    logger.info("Agent question: %s", question)
    final_state = AGENT.invoke(
        {
            "messages": [{"role": "user", "content": question}],
            "stop_reason": "",
            "model_calls": 0,
            "tools_used": [],
        }
    )
    last = final_state["messages"][-1]
    text = "".join(b.text for b in last["content"] if b.type == "text").strip()
    return text or "Sorry, I couldn't finish answering that.", final_state["tools_used"]


DEMO_QUESTIONS = [
    "When is the fridge cleared?",  # needs retrieval only
    "What's 17.5% of 2,480?",  # needs the calculator only
    "I've used up my credit. What would 3 extra hours in the Cedar room cost?",  # both
]


def main() -> int:
    if not has_credentials():
        logger.error(MISSING_CREDENTIALS)
        return 1
    for question in sys.argv[1:] or DEMO_QUESTIONS:
        answer, tools_used = run_agent(question)
        print(f"\nQ: {question}\nA: {answer}\n   (tools used: {', '.join(tools_used) or 'none'})")
    flush_traces()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    sys.exit(main())
