"""Calling the Claude API for one classification, with usage and cost tracking."""

import time

DEFAULT_MODEL = "claude-haiku-5-5"

# USD per million tokens (input, output). Check current prices at
# https://platform.claude.com/docs/en/about-claude/pricing before reporting costs.
PRICES_PER_MTOK = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int):
    if model not in PRICES_PER_MTOK:
        return None
    price_in, price_out = PRICES_PER_MTOK[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def make_client():
    import anthropic  # imported here so the rest of the kit works without it

    # The SDK reads ANTHROPIC_API_KEY from the environment and retries
    # rate-limit and server errors with backoff.
    return anthropic.Anthropic(max_retries=6)


def classify(client, system_prompt: str, user_message: str, model: str = DEFAULT_MODEL,
             max_tokens: int = 64, thinking: bool = False) -> dict:
    """One API call. Returns the reply text plus usage, latency and any error.

    thinking=False (default) turns extended thinking off for short, cheap answers.
    thinking=True lets the model reason first; thinking tokens are billed as output
    and count toward max_tokens, so allow room for them.
    """
    start = time.monotonic()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            thinking={"type": "adaptive"} if thinking else {"type": "disabled"},
            output_config={"effort": "low"},
        )
    except Exception as exc:  # recorded per row so one failure doesn't stop the run
        return {
            "reply": None, "input_tokens": 0, "output_tokens": 0,
            "latency_s": round(time.monotonic() - start, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }
    reply = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    return {
        "reply": reply,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_s": round(time.monotonic() - start, 3),
        "error": None,
    }
