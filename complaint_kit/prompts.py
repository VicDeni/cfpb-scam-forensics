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
Text marked [REDACTED] was removed for privacy.{rules}{examples}

Reply with JSON only, in this exact form: {{"label": "<category>"}}"""

USER_TEMPLATE = "<complaint>\n{text}\n</complaint>"

# Added in version 2 after error analysis: consumers file by the company a complaint is
# against, while the plain prompt led the model to route by topic (e.g. a collector's
# credit report entry went to credit_reporting). Kept separate so the original prompts,
# and their version hashes, stay exactly as they were.
ROUTING_RULES = """

How to choose between similar categories: consumers file a complaint under the company
and product it is against, not under every topic it mentions.
- debt_collection: the complaint is against a debt collector, debt buyer or collection
  agency, including disputes about how that collector reports the debt on a credit report.
- credit_reporting: the complaint is against a credit bureau (Equifax, Experian, TransUnion)
  or a background or tenant screening company, about what a report shows.
- debt_management: the complaint is about a company the consumer paid or signed up with to
  repair credit, settle or reduce debts, get student loan relief or modify a mortgage: its
  fees, promises or results.
- If a lender, card issuer or bank is the target (a disputed charge, a payment, a closed
  account), choose that product even if the complaint also mentions a credit report."""


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


def build_system_prompt(examples=None, rules: bool = False) -> str:
    return SYSTEM_TEMPLATE.format(
        categories=_categories_block(),
        rules=ROUTING_RULES if rules else "",
        examples=_examples_block(examples),
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
