"""Shared LLM plumbing: settings from .env, the Claude client, retries and tracing.

Every call to Claude in this project goes through call_claude(), so retry-with-backoff,
refusal handling and Langfuse cost tracking live in one place.
"""

import functools
import logging
import os
import random
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import anthropic
from dotenv import load_dotenv
from langfuse import Langfuse, get_client, observe

# Read API keys from .env before anything creates a client. Keys never go in the code.
load_dotenv()

logger = logging.getLogger(__name__)

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
# Answering from a few retrieved paragraphs is routine work, so low effort keeps it
# fast and cheap. The judge overrides this with "medium" because grading needs more care.
DEFAULT_EFFORT = os.getenv("CLAUDE_EFFORT", "low")
MAX_TOKENS = 16000

# If Claude's safety classifiers decline a request by mistake, the API re-runs it on
# Anthropic's recommended fallback model instead of returning a refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# US dollars per million tokens (input, output), used to report cost to Langfuse.
PRICES_PER_MILLION = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

def _only_our_spans(span: Any) -> bool:
    """Export only the spans our @observe decorators create.

    Langfuse also records the Anthropic SDK's own spans automatically. Keeping those
    would show every Claude call twice and double the cost in the dashboard, and
    their cost leaves out thinking tokens, which are billed as output.
    """
    scope = span.instrumentation_scope
    return scope is not None and scope.name == "langfuse-sdk"


if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    # The first client created becomes the one get_client() and @observe use.
    Langfuse(should_export_span=_only_our_spans)
else:
    # Without keys, switch tracing off quietly instead of warning on every run.
    os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
    logger.info("Langfuse keys not set, so tracing is off")

P = ParamSpec("P")
R = TypeVar("R")


def is_retryable(error: Exception) -> bool:
    """True for failures that might work if we wait and try again."""
    if isinstance(error, anthropic.APIConnectionError):  # network trouble, includes timeouts
        return True
    if isinstance(error, anthropic.APIStatusError):
        # 429 rate limited, 408/409 timing conflicts, 5xx server errors (529 = overloaded).
        # Anything else (400 bad request, 401 bad key, 404 wrong model) will fail
        # the same way every time, so retrying only wastes time.
        return error.status_code in (408, 409, 429) or error.status_code >= 500
    return False


def with_retry(
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    should_retry: Callable[[Exception], bool] = is_retryable,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry a function with exponential backoff: wait ~1s, ~2s, ~4s ... between tries.

    The random jitter stops many clients that failed together from all retrying at
    the same instant. `sleep` can be swapped out so tests don't actually wait.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    if attempt == max_attempts or not should_retry(error):
                        raise
                    delay = min(base_delay * 2 ** (attempt - 1), max_delay)
                    delay *= random.uniform(0.5, 1.0)
                    logger.warning(
                        "%s failed (%s), retry %d/%d in %.1fs",
                        func.__name__,
                        type(error).__name__,
                        attempt,
                        max_attempts - 1,
                        delay,
                    )
                    sleep(delay)
            raise AssertionError("unreachable")  # the loop always returns or raises

        return wrapper

    return decorator


def has_credentials() -> bool:
    """True if the SDK will find credentials: an API key, a token, or an `ant auth login` profile."""
    profile_dir = os.path.join(os.path.expanduser("~"), ".config", "anthropic")
    return bool(
        os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN") or os.path.isdir(profile_dir)
    )


MISSING_CREDENTIALS = (
    "No Anthropic credentials found. Copy .env.example to .env and set ANTHROPIC_API_KEY."
)


@functools.cache
def get_anthropic_client() -> anthropic.Anthropic:
    # max_retries=0 turns off the SDK's own retries so with_retry is the single
    # place that decides when to retry. Otherwise the two would stack.
    return anthropic.Anthropic(max_retries=0)


def cost_in_dollars(model: str, input_tokens: int, output_tokens: int) -> dict[str, float] | None:
    prices = PRICES_PER_MILLION.get(model)
    if prices is None:
        return None
    return {
        "input": input_tokens * prices[0] / 1_000_000,
        "output": output_tokens * prices[1] / 1_000_000,
    }


@with_retry()
def _create_message(**params: Any) -> Any:
    return get_anthropic_client().beta.messages.create(
        betas=[FALLBACK_BETA], fallbacks="default", **params
    )


@observe(name="claude", as_type="generation")
def call_claude(
    messages: list[dict[str, Any]],
    system: str,
    tools: list[dict[str, Any]] | None = None,
    effort: str = DEFAULT_EFFORT,
    output_format: dict[str, Any] | None = None,
) -> Any:
    """Send one request to Claude and return the full response message.

    Raises RuntimeError if the request is refused, so callers never mistake an
    empty refusal for a real answer.
    """
    output_config: dict[str, Any] = {"effort": effort}
    if output_format is not None:
        output_config["format"] = output_format
    params: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": messages,
        "output_config": output_config,
    }
    if tools:
        params["tools"] = tools

    started = time.perf_counter()
    response = _create_message(**params)
    elapsed = time.perf_counter() - started

    usage = response.usage
    logger.info(
        "Claude (%s) replied in %.1fs: %d input + %d output tokens, stop_reason=%s",
        response.model,
        elapsed,
        usage.input_tokens,
        usage.output_tokens,
        response.stop_reason,
    )
    get_client().update_current_generation(
        model=response.model,
        input={"system": system, "messages": messages},
        output=[block.to_dict() for block in response.content],
        usage_details={"input": usage.input_tokens, "output": usage.output_tokens},
        cost_details=cost_in_dollars(response.model, usage.input_tokens, usage.output_tokens),
        metadata={"stop_reason": response.stop_reason, "effort": effort},
    )

    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise RuntimeError(f"Claude declined this request (category: {category})")
    return response


def response_text(response: Any) -> str:
    """Join the text blocks of a response, skipping thinking and tool-use blocks."""
    return "".join(block.text for block in response.content if block.type == "text").strip()


def flush_traces() -> None:
    """Send any buffered traces to Langfuse before the program exits."""
    get_client().flush()
