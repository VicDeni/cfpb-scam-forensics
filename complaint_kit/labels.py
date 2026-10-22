"""Product taxonomy: map CFPB's changing product names onto stable classes.

The CFPB renamed and merged product categories over the years (for example
"Bank account or service" became "Checking or savings account"). A model needs
one consistent label set, so every raw product string goes through map_product.
"""

import re

LABELS = {
    "credit_reporting": "Credit reports: errors, disputes, identity theft on a "
    "report, credit monitoring or credit repair services",
    "debt_collection": "Debt collectors: attempts to collect a debt, "
    "communication tactics, debt not owed, threats",
    "credit_card": "Credit card accounts: billing disputes, fees, interest, "
    "rewards, card fraud, card applications",
    "checking_savings": "Checking or savings accounts: deposits, withdrawals, "
    "overdraft fees, opening or closing an account, CDs",
    "mortgage": "Home mortgages: servicing, payments, escrow, foreclosure, "
    "loan modification, applying for a mortgage",
    "student_loan": "Federal or private student loans: servicing, repayment, "
    "forgiveness",
    "vehicle_loan": "Auto loans or leases: payments, repossession, servicing",
    "personal_loan": "Payday, title, installment or personal loans, or cash advances",
    "money_transfer": "Money transfers, payment apps, wires, virtual currency, "
    "check cashing, money orders",
    "prepaid_card": "Prepaid debit cards, gift cards, government benefit cards",
}

EXCLUDED = "EXCLUDED"  # known category, but too ambiguous to train on
UNMAPPED = "UNMAPPED"  # unknown category: add a rule below

# Checked in order; first match wins. Patterns match the lowercased product.
RULES = [
    (r"credit reporting|credit repair", "credit_reporting"),
    (r"^debt collection", "debt_collection"),
    # 2017-2022 category merged cards and prepaid; credit cards dominate it.
    (r"^credit card or prepaid card$", "credit_card"),
    (r"^credit card$", "credit_card"),
    (r"prepaid card", "prepaid_card"),
    (r"checking or savings|bank account or service", "checking_savings"),
    (r"mortgage", "mortgage"),
    (r"student loan", "student_loan"),
    (r"vehicle loan", "vehicle_loan"),
    (r"payday loan|personal loan", "personal_loan"),
    (r"money transfer|virtual currency|money service", "money_transfer"),
    # Pre-2017 "Consumer Loan" mixed auto and installment loans.
    (r"^consumer loan$", EXCLUDED),
    (r"^other financial service$", EXCLUDED),
]

_COMPILED = [(re.compile(p), label) for p, label in RULES]


def map_product(product) -> str:
    if product is None or not str(product).strip():
        return UNMAPPED
    text = str(product).strip().lower()
    for pattern, label in _COMPILED:
        if pattern.search(text):
            return label
    return UNMAPPED


def normalize_label(text) -> str | None:
    """Turn a model's answer ('Credit card', 'credit_card') into a label key."""
    if text is None:
        return None
    key = re.sub(r"[^a-z]+", "_", str(text).lower()).strip("_")
    return key if key in LABELS else None
