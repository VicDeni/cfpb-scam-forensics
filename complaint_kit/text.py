"""PII scanning/masking and narrative cleaning.

The CFPB already redacts names, account numbers and dates as XXXX before
publishing, but redaction is never perfect. We check for anything left over and
mask it before any text is used for modeling or sent to an external LLM API.

Patterns are written to work in both Python `re` and Java regex, so the same
strings drive pandas code here and Spark `rlike`/`regexp_replace` in Databricks.
"""

import re

# Order matters for masking: more specific patterns first.
PII_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "card": r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{1,4}\b",
    "phone": r"(\(\d{3}\) ?|\b\d{3}[-. ])\d{3}[-. ]\d{4}\b",
    "long_number": r"\b\d{9,19}\b",
}

PII_TOKENS = {name: f"[{name.upper()}]" for name in PII_PATTERNS}

# CFPB redaction marks: XXXX, XX/XX/XXXX, XXXX-XXXX ...
REDACTION_PATTERN = r"\bX{2,}(?:[/-]X{2,})*\b"
REDACTED_RUN_PATTERN = r"(?:\[REDACTED\]\s*){2,}"
BRACES_PATTERN = r"[{}]"  # CFPB wraps dollar amounts like {$500.00}
WHITESPACE_PATTERN = r"\s+"

_PII = {k: re.compile(v) for k, v in PII_PATTERNS.items()}


def find_pii(text: str) -> dict:
    """Which PII types appear in the text, e.g. {'email': True, 'ssn': False, ...}."""
    text = text or ""
    return {name: bool(p.search(text)) for name, p in _PII.items()}


def mask_pii(text: str) -> str:
    text = text or ""
    for name, p in _PII.items():
        text = p.sub(PII_TOKENS[name], text)
    return text


def clean_narrative(text: str) -> str:
    """Mask PII, collapse XXXX redactions, drop braces, normalize whitespace."""
    text = mask_pii(text)
    text = re.sub(REDACTION_PATTERN, "[REDACTED]", text)
    text = re.sub(REDACTED_RUN_PATTERN, "[REDACTED] ", text)
    text = re.sub(BRACES_PATTERN, "", text)
    return re.sub(WHITESPACE_PATTERN, " ", text).strip()


def truncate(text: str, max_chars: int) -> str:
    """Cut long narratives at a word boundary to bound LLM cost."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " [TRUNCATED]"
