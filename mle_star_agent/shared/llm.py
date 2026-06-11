"""LiteLLM call wrapper with token tracking, model switching, and retry logic.

Replaces ADK shared/callbacks.py. Provides:
- Synchronous and async LiteLLM call wrappers
- Model selection: MODEL_PRO normally, MODEL_FLASH once TOKEN_LITE_THRESHOLD is passed
- Built-in rate-limit retry with exponential backoff (handles 429/503/quota errors)
- Token budget stop: raises TokenBudgetExceeded when tokens_used >= TOKEN_BUDGET
- Prompt builder: builds standard system/user messages list
- Structured JSON response parser: extracts JSON from LLM response text
- Token counter: module-level counter reset per graph run; per-call dict update for state
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

import litellm

from mle_star_agent import config

logger = logging.getLogger(__name__)

# Module-level cumulative token counter. Reset via reset_token_count() at the
# start of each graph run (or retry cycle — mirrors ADK retry_loop_agent reset).
_token_count: int = 0


class TokenBudgetExceeded(RuntimeError):
    """Raised when the cumulative token count reaches TOKEN_BUDGET."""


def reset_token_count() -> None:
    """Reset the module-level token counter to zero (call at graph/retry start)."""
    global _token_count
    _token_count = 0


def get_token_count() -> int:
    """Return the current module-level token count."""
    return _token_count


def _select_model(current_tokens: int) -> str:
    """Return MODEL_FLASH once above TOKEN_LITE_THRESHOLD, else MODEL_PRO."""
    if current_tokens >= config.TOKEN_LITE_THRESHOLD:
        logger.debug(
            "Token lite threshold reached (%d >= %d): using MODEL_FLASH",
            current_tokens, config.TOKEN_LITE_THRESHOLD,
        )
        return config.MODEL_FLASH
    return config.MODEL_PRO


def build_messages(system: str, user: str) -> list[dict]:
    """Build a standard two-message list for a single LLM call."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_json_response(text: str) -> Any:
    """Extract the first JSON object or array from an LLM response string.

    Handles markdown code fences (```json ... ```) and bare JSON objects/arrays.
    Raises ValueError when no valid JSON is found.
    """
    # Strip common markdown code fences
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
    stripped = re.sub(r"\n?```$", "", stripped.strip())
    stripped = stripped.strip()

    # Try direct parse first (covers bare JSON)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Walk through the string to find a balanced JSON object or array. The model
    # may emit prose containing stray braces before the real JSON, so on a failed
    # parse we keep scanning from the next opening delimiter rather than giving up.
    for start_char, end_char in (("{", "}"), ("[", "]")):
        search_from = 0
        while True:
            idx = stripped.find(start_char, search_from)
            if idx == -1:
                break
            depth = 0
            in_string = False
            escape = False
            matched_end = -1
            for pos in range(idx, len(stripped)):
                ch = stripped[pos]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        matched_end = pos
                        break
            if matched_end != -1:
                try:
                    return json.loads(stripped[idx:matched_end + 1])
                except json.JSONDecodeError:
                    pass
            # Advance past this opening delimiter and try the next candidate span.
            search_from = idx + 1

    raise ValueError(f"No valid JSON found in LLM response: {text[:300]!r}")


def _update_token_state(used: int, token_state: Optional[dict]) -> None:
    """Add `used` tokens to the module counter and optionally to a state dict."""
    global _token_count
    _token_count += used
    if token_state is not None:
        token_state["token_count"] = token_state.get("token_count", 0) + used


def _check_budget(current_tokens: int) -> None:
    if current_tokens >= config.TOKEN_BUDGET:
        raise TokenBudgetExceeded(
            f"Token budget exhausted: {current_tokens} >= {config.TOKEN_BUDGET}. "
            "Set stop_outer_loop=True and wind down."
        )


def call_llm(
    messages: list[dict],
    *,
    model: Optional[str] = None,
    force_flash: bool = False,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    token_state: Optional[dict] = None,
) -> str:
    """Synchronous LiteLLM call with retry, token tracking, and budget enforcement.

    Args:
        messages: Chat messages list (build via ``build_messages``).
        model: Explicit model override; if None, auto-selects based on token budget.
        force_flash: When True, always use MODEL_FLASH regardless of budget.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
        token_state: Mutable dict with optional ``token_count`` key for per-run
            state tracking (mirrors ADK state["token_count"]).

    Returns:
        Response text content string.

    Raises:
        TokenBudgetExceeded: When cumulative tokens reach TOKEN_BUDGET.
    """
    current_tokens = (token_state or {}).get("token_count", _token_count)
    _check_budget(current_tokens)

    if model is None:
        model = config.MODEL_FLASH if force_flash else _select_model(current_tokens)

    api_key = config.require_api_key()
    max_retries = 4
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=api_key,
            )
            used = 0
            if hasattr(response, "usage") and response.usage:
                used = int(response.usage.total_tokens or 0)
            _update_token_state(used, token_state)
            logger.debug("LLM: model=%s used=%d cumulative=%d", model, used, _token_count)

            if _token_count >= config.TOKEN_BUDGET:
                logger.warning(
                    "Token budget reached after call: %d >= %d",
                    _token_count, config.TOKEN_BUDGET,
                )

            content = response.choices[0].message.content or ""
            if not content:
                logger.warning(
                    "LLM returned empty content (model=%s max_tokens=%d). "
                    "Extended-thinking may have consumed all token budget — "
                    "raise SCRIPT_MAX_TOKENS if this happens on script-gen calls.",
                    model, max_tokens,
                )
            return content

        except Exception as exc:
            if _is_transient(exc) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Transient LLM error (attempt %d/%d, %.1fs retry): %s",
                    attempt + 1, max_retries, delay, exc,
                )
                time.sleep(delay)
                continue
            raise

    raise RuntimeError("LLM call failed after all retries")


async def acall_llm(
    messages: list[dict],
    *,
    model: Optional[str] = None,
    force_flash: bool = False,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    token_state: Optional[dict] = None,
) -> str:
    """Async version of ``call_llm`` for use inside LangGraph async nodes."""
    import asyncio

    current_tokens = (token_state or {}).get("token_count", _token_count)
    _check_budget(current_tokens)

    if model is None:
        model = config.MODEL_FLASH if force_flash else _select_model(current_tokens)

    api_key = config.require_api_key()
    max_retries = 4
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=api_key,
            )
            used = 0
            if hasattr(response, "usage") and response.usage:
                used = int(response.usage.total_tokens or 0)
            _update_token_state(used, token_state)
            logger.debug("LLM async: model=%s used=%d cumulative=%d", model, used, _token_count)

            if _token_count >= config.TOKEN_BUDGET:
                logger.warning(
                    "Token budget reached after async call: %d >= %d",
                    _token_count, config.TOKEN_BUDGET,
                )

            content = response.choices[0].message.content or ""
            if not content:
                logger.warning(
                    "LLM returned empty content async (model=%s max_tokens=%d). "
                    "Extended-thinking may have consumed all token budget — "
                    "raise SCRIPT_MAX_TOKENS if this happens on script-gen calls.",
                    model, max_tokens,
                )
            return content

        except Exception as exc:
            if _is_transient(exc) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Transient LLM error async (attempt %d/%d, %.1fs retry): %s",
                    attempt + 1, max_retries, delay, exc,
                )
                await asyncio.sleep(delay)
                continue
            raise

    raise RuntimeError("LLM async call failed after all retries")


def call_llm_json(
    messages: list[dict],
    *,
    model: Optional[str] = None,
    force_flash: bool = False,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    token_state: Optional[dict] = None,
) -> Any:
    """Convenience wrapper: call_llm + parse_json_response."""
    text = call_llm(
        messages,
        model=model,
        force_flash=force_flash,
        max_tokens=max_tokens,
        temperature=temperature,
        token_state=token_state,
    )
    return parse_json_response(text)


async def acall_llm_json(
    messages: list[dict],
    *,
    model: Optional[str] = None,
    force_flash: bool = False,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    token_state: Optional[dict] = None,
) -> Any:
    """Async convenience wrapper: acall_llm + parse_json_response."""
    text = await acall_llm(
        messages,
        model=model,
        force_flash=force_flash,
        max_tokens=max_tokens,
        temperature=temperature,
        token_state=token_state,
    )
    return parse_json_response(text)


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in (
        "503", "429", "rate limit", "too many requests",
        "resource_exhausted", "quota", "overloaded",
        "service unavailable", "gateway timeout",
    ))
