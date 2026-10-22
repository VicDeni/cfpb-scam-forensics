"""Prompt templates, versioning and reply parsing for LLM classification.

Every prompt gets a version string: its name plus a short hash of the exact
text (including any few-shot examples). Editing a prompt changes its version
automatically, so results can always be traced to the prompt that made them.
"""

import hashlib
import json
import re

from .labels import LABELS, normalize_label

PARSE_ERROR = "PARSE_ERROR"


def _categories_block() -> str:
    return "\n".join(f"- {key}: {desc}" for key, desc in LABELS.items())


SYSTEM_TEMPLATE = """You route consumer complaints to the right product team at a bank.
Read the complaint and choose exactly one product category from this list:

{categories}

Pick the product the complaint is mainly about. If a debt collector is
collecting on a credit card or loan, the category is debt_collection.
Text marked [REDACTED] was removed for privacy.{examples}

Reply with JSON only, in this exact form: {{"label": "<category>"}}"""

USER_TEMPLATE = "<complaint>\n{text}\n</complaint>"


def _examples_block(examples) -> str:
    if not examples:
        return ""
    parts = ["\n\nExamples:"]
    for ex in examples:
        parts.append(
            f"\n<complaint>\n{ex['text']}\n</complaint>\n"
            f'{{"label": "{ex["label"]}"}}'
        )
    return "\n".join(parts)


def build_system_prompt(examples=None) -> str:
    return SYSTEM_TEMPLATE.format(
        categories=_categories_block(), examples=_examples_block(examples)
    )


def build_user_message(text: str) -> str:
    return USER_TEMPLATE.format(text=text)


def prompt_version(name: str, system_prompt: str) -> str:
    digest = hashlib.sha256((system_prompt + USER_TEMPLATE).encode()).hexdigest()
    return f"{name}-{digest[:8]}"


def parse_label(reply: str) -> str:
    """Extract a label from the model's reply, or PARSE_ERROR."""
    if not reply:
        return PARSE_ERROR
    match = re.search(r"\{.*?\}", reply, re.DOTALL)
    if match:
        try:
            label = normalize_label(json.loads(match.group(0)).get("label"))
            if label:
                return label
        except (json.JSONDecodeError, AttributeError):
            pass
    # Fallback: the reply names exactly one valid label somewhere.
    found = {key for key in LABELS if re.search(rf"\b{key}\b", reply.lower())}
    return found.pop() if len(found) == 1 else PARSE_ERROR
