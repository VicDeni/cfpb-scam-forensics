"""Payment-scam extraction: field definitions, prompt, reply validation, scoring.

The same definitions drive the LLM prompt and the human labeling guide
(docs/labeling_guide.md), so the model and the human labeler follow the same rules.
"""

import hashlib
import json
import math
import re

PARSE_ERROR = "PARSE_ERROR"
NA = "not_applicable"

# Products where payment scams show up, and words that suggest one.
SCAM_PRODUCTS = ["money_transfer", "checking_savings", "prepaid_card"]
SCAM_KEYWORDS = (r"(?i)\b(scam|scammer|scammed|fraud|fraudster|fraudulent|zelle|wire transfer|"
                 r"impersonat|bitcoin|crypto|cash app|venmo|gift card)")

FIELDS = {
    "fraud_type": {
        "question": "What kind of problem is this?",
        "values": {
            "authorized_scam": "The consumer was tricked into sending the money or approving the "
            "payment themselves, e.g. someone posing as their bank told them to move money to a "
            "'safe account', or they paid a fake seller.",
            "unauthorized_transaction": "Money left the account without the consumer's action or "
            "approval: stolen card or phone, hacked account, a payment they did not make.",
            "not_fraud": "No fraud or scam is described, e.g. a fee dispute, a delayed or stuck "
            "transfer, or a disagreement with a real merchant.",
        },
    },
    "scam_type": {
        "question": "If authorized_scam: what was the story the scammer used?",
        "values": {
            "bank_impersonation": "Posed as the consumer's bank or its fraud department.",
            "government_impersonation": "Posed as a government agency (IRS, Social Security, "
            "police, courts).",
            "business_impersonation": "Posed as a well-known company or tech support (Amazon, "
            "Apple, Microsoft, a utility).",
            "romance": "An online relationship used to ask for money.",
            "investment_crypto": "A fake investment, trading or crypto opportunity.",
            "purchase_or_sale": "A fake seller or buyer: marketplace item, rental, tickets, pets.",
            "job_or_income": "A fake job, overpayment, or money-making scheme.",
            "family_or_friend": "Posed as a relative or friend in an emergency.",
            "other": "A scam that fits none of the above.",
            "unclear": "A scam, but not enough detail to tell which kind.",
            NA: "Not an authorized_scam.",
        },
    },
    "contact_channel": {
        "question": "If authorized_scam: how did the scammer first reach the consumer?",
        "values": {
            "phone_call": "Phone call.",
            "text_message": "Text message.",
            "email": "Email.",
            "social_media": "Social media or messaging app.",
            "dating_site": "Dating app or site.",
            "online_marketplace": "Online marketplace or classifieds listing.",
            "website_or_ad": "A website, search result or online ad.",
            "other": "Some other way (in person, mail).",
            "unclear": "Not stated.",
            NA: "Not an authorized_scam.",
        },
    },
    "payment_method": {
        "question": "If fraud: how did the money leave?",
        "values": {
            "zelle": "Zelle.",
            "other_p2p_app": "Another payment app: Cash App, Venmo, PayPal, Apple Cash.",
            "wire_transfer": "Bank wire transfer.",
            "crypto": "Cryptocurrency, including crypto ATMs.",
            "card": "Debit or credit card payment.",
            "bank_transfer": "ACH or other account-to-account transfer, or online bill pay.",
            "check": "Check or money order.",
            "gift_card": "Gift cards.",
            "cash_withdrawal": "Cash withdrawn and handed over or deposited.",
            "multiple": "More than one of these.",
            "unclear": "Not stated.",
            NA: "Not fraud.",
        },
    },
    "refund_status": {
        "question": "If fraud: what had the bank or payment company done about the loss "
        "when the consumer wrote the complaint?",
        "values": {
            "refunded_full": "Refunded all of it.",
            "refunded_partial": "Refunded some of it.",
            "denied": "Refused to refund, or said the consumer is responsible.",
            "pending": "Still investigating, or the consumer is waiting.",
            "not_mentioned": "The complaint doesn't say.",
            NA: "Not fraud.",
        },
    },
}
AMOUNT_FIELD = "amount_lost_usd"
AMOUNT_RULE = ("Total US dollars the consumer says they lost or sent to the fraudster, as a "
               "number (e.g. 1250.00). Add up multiple payments. Leave empty (null) if no amount "
               "is stated or it isn't fraud.")
ALL_FIELDS = list(FIELDS) + [AMOUNT_FIELD]

RULES = [
    "If fraud_type is not_fraud, every other field is not_applicable and the amount is null.",
    "If fraud_type is unauthorized_transaction, scam_type and contact_channel are not_applicable.",
    "Judge only from what the consumer wrote. Don't assume details that aren't there; use unclear "
    "or not_mentioned instead.",
    "Text marked [REDACTED] was removed for privacy; [PHONE], [EMAIL] and similar are masked values.",
]


# ---------- prompt ----------

def _fields_block() -> str:
    lines = []
    for name, spec in FIELDS.items():
        lines.append(f"{name}: {spec['question']}")
        lines += [f"  - {value}: {desc}" for value, desc in spec["values"].items()]
    lines.append(f"{AMOUNT_FIELD}: {AMOUNT_RULE}")
    return "\n".join(lines)


def build_system_prompt() -> str:
    example = {name: "<value>" for name in FIELDS} | {AMOUNT_FIELD: "<number or null>"}
    rules = "\n".join(f"- {r}" for r in RULES)
    return (
        "You analyze consumer complaints about banks and payment apps to understand fraud and "
        "scams. Read the complaint and fill in these fields, choosing values only from the lists.\n\n"
        f"{_fields_block()}\n\nRules:\n{rules}\n\n"
        f"Reply with JSON only, in this exact form: {json.dumps(example)}"
    )


def build_user_message(text: str) -> str:
    return f"<complaint>\n{text}\n</complaint>"


def prompt_version(system_prompt: str) -> str:
    return "scam_extract-" + hashlib.sha256(system_prompt.encode()).hexdigest()[:8]


# ---------- validating values (used for model replies and human labels) ----------

def normalize_value(field: str, value):
    """Map a raw value to an allowed one. None for blank, PARSE_ERROR if invalid."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    key = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if not key:
        return None
    return key if key in FIELDS[field]["values"] else PARSE_ERROR


def parse_amount(value):
    """'$1,250.00' -> 1250.0. Returns (amount or None, ok)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, True
    if isinstance(value, (int, float)):
        return (float(value), True) if value >= 0 else (None, False)
    text = str(value).strip().lower()
    if text in ("", "null", "none", "n/a"):
        return None, True
    text = text.replace("$", "").replace(",", "")
    try:
        amount = float(text)
    except ValueError:
        return None, False
    return (amount, True) if amount >= 0 else (None, False)


def parse_reply(reply: str) -> dict:
    """Validate a model reply. Bad or missing fields become PARSE_ERROR."""
    out = {name: PARSE_ERROR for name in FIELDS} | {AMOUNT_FIELD: None, "amount_ok": False}
    if not reply:
        return out
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return out
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return out
    if not isinstance(data, dict):
        return out
    for name in FIELDS:
        out[name] = normalize_value(name, data.get(name)) or PARSE_ERROR
    out[AMOUNT_FIELD], out["amount_ok"] = parse_amount(data.get(AMOUNT_FIELD))
    return out


def is_consistent(row: dict) -> bool:
    """Do the fields follow the RULES above?"""
    fraud = row["fraud_type"]
    if fraud == "not_fraud":
        return all(row[f] == NA for f in FIELDS if f != "fraud_type") and row[AMOUNT_FIELD] is None
    if fraud == "unauthorized_transaction":
        return row["scam_type"] == NA and row["contact_channel"] == NA
    return True


# ---------- scoring against human labels ----------

def wilson_interval(correct: int, n: int, z: float = 1.96):
    """95% confidence interval for an accuracy measured on n examples."""
    if n == 0:
        return (None, None)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def amounts_match(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= max(1.0, 0.01 * abs(b))


def score(human_rows: list, model_rows: dict):
    """Compare human labels to model output, field by field.

    human_rows: dicts from the label sheet. model_rows: {complaint_id: parsed model row}.
    Blank human cells are skipped, so a partly labeled sheet still works.
    Returns (per_field results, disagreements, invalid human labels).
    """
    tallies = {f: [0, 0] for f in ALL_FIELDS}  # field -> [correct, n]
    disagreements, invalid = [], []
    for h in human_rows:
        cid = str(h["complaint_id"])
        m = model_rows.get(cid)
        if m is None:
            continue
        for field in FIELDS:
            want = normalize_value(field, h.get(field))
            if want is None:
                continue
            if want == PARSE_ERROR:
                invalid.append({"complaint_id": cid, "field": field, "value": h.get(field)})
                continue
            ok = m[field] == want
            tallies[field][0] += ok
            tallies[field][1] += 1
            if not ok:
                disagreements.append({"complaint_id": cid, "field": field,
                                      "human": want, "model": m[field]})
        raw_amount = h.get(AMOUNT_FIELD)
        if not _blank(raw_amount):  # "none" means no amount; blank means not labeled
            amount, ok_amount = parse_amount(raw_amount)
            if not ok_amount:
                invalid.append({"complaint_id": cid, "field": AMOUNT_FIELD, "value": raw_amount})
            else:
                ok = amounts_match(m[AMOUNT_FIELD], amount)
                tallies[AMOUNT_FIELD][0] += ok
                tallies[AMOUNT_FIELD][1] += 1
                if not ok:
                    disagreements.append({"complaint_id": cid, "field": AMOUNT_FIELD,
                                          "human": amount, "model": m[AMOUNT_FIELD]})
    results = []
    for field, (correct, n) in tallies.items():
        low, high = wilson_interval(correct, n)
        results.append({"field": field, "n": n, "accuracy": correct / n if n else None,
                        "ci_low": low, "ci_high": high})
    return results, disagreements, invalid


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == ""


# ---------- labeling guide ----------

def labeling_guide_markdown() -> str:
    lines = [
        "# Labeling guide: payment-scam complaints",
        "",
        "Generated from `complaint_kit/scam.py`. The LLM gets exactly these definitions, so "
        "you and the model are judged by the same rules.",
        "",
        "**Label every row before you look at any model output.** Seeing the model's answer "
        "first biases you toward agreeing with it, which makes its accuracy look better than it is.",
        "",
        "The easiest way to label is `python scripts/label_scams.py`, which shows one complaint "
        "at a time, lets you pick each value by number, applies the rules below for you, and "
        "saves as you go. If you edit the sheet by hand instead, type values exactly as written "
        "below. Blank cells are skipped when scoring. Use `notes` for anything ambiguous: those "
        "cases are good material for the error analysis.",
        "",
        "## Rules",
        "",
        *[f"- {r}" for r in RULES],
        "",
    ]
    for name, spec in FIELDS.items():
        lines += [f"## `{name}`", "", spec["question"], "", "| Value | Meaning |", "|---|---|"]
        lines += [f"| `{v}` | {d} |" for v, d in spec["values"].items()]
        lines.append("")
    lines += [f"## `{AMOUNT_FIELD}`", "", AMOUNT_RULE, "",
              "In the sheet, type `none` when no amount is stated (a blank cell means you "
              "skipped it). Plain numbers or `$1,250.00` both work.", ""]
    return "\n".join(lines)
