import re
from types import SimpleNamespace

import pandas as pd
import pytest

from complaint_kit import evaluate, labels, llm, prompts, sampling, schema, text


# ---------- schema ----------

@pytest.mark.parametrize("raw, expected", [
    ("Date received", "date_received"),
    ("Sub-product", "sub_product"),
    ("Consumer consent provided?", "consumer_consent_provided"),
    ("Timely response?", "timely_response"),
    ("ZIP code", "zip_code"),
    ("Complaint ID", "complaint_id"),
    ("complaint_what_happened", "consumer_complaint_narrative"),  # JSON/API name
    ("company_response", "company_response_to_consumer"),
    ("timely", "timely_response"),
])
def test_normalize_column_name(raw, expected):
    assert schema.normalize_column_name(raw) == expected


def test_missing_required():
    assert schema.missing_required(schema.REQUIRED_COLUMNS) == []
    assert schema.missing_required(["complaint_id"]) == schema.REQUIRED_COLUMNS[1:]


def test_data_dictionary_markdown_lists_every_table():
    md = schema.data_dictionary_markdown()
    for table in schema.DATA_DICTIONARY:
        assert f"## `{table}`" in md


# ---------- labels ----------

@pytest.mark.parametrize("product, expected", [
    ("Credit reporting, credit repair services, or other personal consumer reports", "credit_reporting"),
    ("Credit reporting or other personal consumer reports", "credit_reporting"),
    ("Credit reporting", "credit_reporting"),
    ("Debt collection", "debt_collection"),
    ("Debt or credit management", "debt_management"),
    ("Credit card or prepaid card", "credit_card"),
    ("Credit card", "credit_card"),
    ("Prepaid card", "prepaid_card"),
    ("Checking or savings account", "checking_savings"),
    ("Bank account or service", "checking_savings"),
    ("Mortgage", "mortgage"),
    ("Student loan", "student_loan"),
    ("Vehicle loan or lease", "vehicle_loan"),
    ("Payday loan, title loan, or personal loan", "personal_loan"),
    ("Payday loan, title loan, personal loan, or advance loan", "personal_loan"),
    ("Payday loan", "personal_loan"),
    ("Money transfer, virtual currency, or money service", "money_transfer"),
    ("Money transfers", "money_transfer"),
    ("Virtual currency", "money_transfer"),
    ("Consumer Loan", labels.EXCLUDED),
    ("Other financial service", labels.EXCLUDED),
    ("Something brand new", labels.UNMAPPED),
    ("", labels.UNMAPPED),
    (None, labels.UNMAPPED),
])
def test_map_product(product, expected):
    assert labels.map_product(product) == expected


@pytest.mark.parametrize("answer, expected", [
    ("credit_card", "credit_card"),
    ("Credit Card", "credit_card"),
    (" debt collection ", "debt_collection"),
    ("banana", None),
    (None, None),
])
def test_normalize_label(answer, expected):
    assert labels.normalize_label(answer) == expected


# ---------- text / PII ----------

def test_find_pii_detects_each_type():
    sample = ("Email me at jane.doe@example.com, SSN 123-45-6789, card 4111 1111 1111 1111, "
              "call (215) 555-0134, account 1234567890123.")
    assert all(text.find_pii(sample).values())


def test_find_pii_ignores_cfpb_redactions_and_amounts():
    sample = "On XX/XX/2025 I paid {$1500.00} to XXXX XXXX on account XXXX1234."
    assert not any(text.find_pii(sample).values())


def test_mask_pii_removes_values():
    masked = text.mask_pii("Reach me at 215-555-0134 or jane@example.com, SSN 123-45-6789")
    assert "555-0134" not in masked and "jane@" not in masked and "6789" not in masked
    assert "[PHONE]" in masked and "[EMAIL]" in masked and "[SSN]" in masked


def test_clean_narrative():
    raw = "On XX/XX/XXXX  XXXX XXXX charged me {$45.00}\n\nin fees. Call 215.555.0134"
    # Runs of redactions collapse into one marker.
    assert text.clean_narrative(raw) == "On [REDACTED] charged me $45.00 in fees. Call [PHONE]"


def test_clean_narrative_keeps_ordinary_words():
    assert text.clean_narrative("Size XXL Xbox") == "Size XXL Xbox"


def test_truncate():
    assert text.truncate("short", 100) == "short"
    out = text.truncate("one two three four five", 12)
    assert out == "one two [TRUNCATED]"


def test_patterns_are_plain_regex():
    # The same strings are used by Spark (Java regex): no Python-only syntax.
    for pattern in list(text.PII_PATTERNS.values()) + [text.REDACTION_PATTERN]:
        assert "(?P<" not in pattern and "\\Z" not in pattern
        re.compile(pattern)


# ---------- prompts ----------

def test_system_prompt_lists_all_labels():
    system = prompts.build_system_prompt()
    for key in labels.LABELS:
        assert f"- {key}:" in system
    assert "Examples:" not in system


def test_few_shot_examples_change_prompt_and_version():
    zero = prompts.build_system_prompt()
    few = prompts.build_system_prompt([{"label": "mortgage", "text": "My escrow went up."}])
    assert "My escrow went up." in few
    assert prompts.prompt_version("p", zero) != prompts.prompt_version("p", few)
    assert prompts.prompt_version("p", zero) == prompts.prompt_version("p", zero)


@pytest.mark.parametrize("reply, expected", [
    ('{"label": "mortgage"}', "mortgage"),
    ('Sure! {"label": "Credit card"}', "credit_card"),
    ('```json\n{"label": "student_loan"}\n```', "student_loan"),
    ("The answer is debt_collection.", "debt_collection"),
    ("credit_card or prepaid_card", prompts.PARSE_ERROR),
    ('{"label": "banana"}', prompts.PARSE_ERROR),
    ("", prompts.PARSE_ERROR),
    (None, prompts.PARSE_ERROR),
])
def test_parse_label(reply, expected):
    assert prompts.parse_label(reply) == expected


# ---------- llm (no network: fake client) ----------

class FakeMessages:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("boom")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"label": "mortgage"}')],
            usage=SimpleNamespace(input_tokens=500, output_tokens=8),
        )


def test_classify_success_and_request_shape():
    client = SimpleNamespace(messages=FakeMessages())
    out = llm.classify(client, "sys", "user")
    assert out["reply"] == '{"label": "mortgage"}' and out["error"] is None
    assert (out["input_tokens"], out["output_tokens"]) == (500, 8)
    call = client.messages.calls[0]
    assert call["model"] == llm.DEFAULT_MODEL
    assert call["thinking"] == {"type": "disabled"}
    assert "temperature" not in call  # not recommended for current models


def test_classify_records_errors_instead_of_raising():
    out = llm.classify(SimpleNamespace(messages=FakeMessages(fail=True)), "sys", "user")
    assert out["reply"] is None and "boom" in out["error"]


def test_estimate_cost():
    assert llm.estimate_cost("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert llm.estimate_cost("unknown-model", 10, 10) is None


# ---------- evaluate ----------

def test_summarize_counts_parse_errors_as_wrong():
    y_true = ["mortgage", "credit_card", "debt_collection", "mortgage"]
    y_pred = ["mortgage", "credit_card", prompts.PARSE_ERROR, "credit_card"]
    out = evaluate.summarize(y_true, y_pred, [100] * 4, [5] * 4, [0.001] * 4)
    assert out["accuracy"] == 0.5
    assert out["parse_error_rate"] == 0.25
    assert out["cost_per_1k_usd"] == pytest.approx(1.0)
    assert out["avg_input_tokens"] == 100


def test_per_class_and_confusions():
    y_true = ["mortgage", "mortgage", "credit_card"]
    y_pred = ["mortgage", "credit_card", "credit_card"]
    rows = {r["label"]: r for r in evaluate.per_class(y_true, y_pred)}
    assert rows["mortgage"]["recall"] == 0.5 and rows["credit_card"]["precision"] == 0.5
    assert evaluate.top_confusions(y_true, y_pred) == [
        {"true_label": "mortgage", "predicted": "credit_card", "count": 1}
    ]


# ---------- sampling ----------

def test_stratified_sample_caps_each_class():
    df = pd.DataFrame({"label": ["a"] * 50 + ["b"] * 3, "x": range(53)})
    out = sampling.stratified_sample(df, "label", per_class=10)
    assert out["label"].value_counts().to_dict() == {"a": 10, "b": 3}
    pd.testing.assert_frame_equal(out, sampling.stratified_sample(df, "label", per_class=10))


def test_pick_examples_respects_length_and_classes():
    df = pd.DataFrame({
        "complaint_id": range(6),
        "label": ["a", "a", "b", "b", "c", "c"],
        "text": ["x" * 300, "x" * 900, "y" * 250, "y" * 50, "z" * 10, "z" * 2000],
    })
    examples = {e["label"]: e["text"] for e in sampling.pick_examples(df, "label", "text")}
    assert examples["a"] == "x" * 300 and examples["b"] == "y" * 250  # in-range picks
    # "c" has nothing in range, so it falls back to a narrative cut to max_chars
    assert examples["c"] in ("z" * 10, "z" * 600 + " [TRUNCATED]")


def test_stratified_sample_empty():
    assert sampling.stratified_sample(pd.DataFrame({"label": []}), "label", 5).empty


def test_macro_f1_ignores_classes_absent_from_truth():
    out = evaluate.summarize(["mortgage", "mortgage"], ["mortgage", "mortgage"])
    assert out["macro_f1"] == 1.0


# ---------- governance / docs ----------

def test_data_dictionary_doc_is_up_to_date():
    # If this fails: python -c "from complaint_kit.schema import data_dictionary_markdown as m; open('docs/data_dictionary.md','w').write(m())"
    from pathlib import Path
    doc = Path(__file__).resolve().parents[1] / "docs" / "data_dictionary.md"
    assert doc.read_text() == schema.data_dictionary_markdown()


def test_comment_statements_escape_quotes_and_skip_missing_columns():
    from complaint_kit.governance import comment_statements
    stmts = comment_statements("c.s.silver_complaints", "silver_complaints", ["complaint_id"])
    assert stmts[0].startswith("COMMENT ON TABLE c.s.silver_complaints IS '")
    assert len(stmts) == 2 and "`complaint_id`" in stmts[1]
    schema.DATA_DICTIONARY["_t"] = ("It's quoted", {})
    try:
        assert "It\\'s quoted" in comment_statements("x", "_t", [])[0]
    finally:
        del schema.DATA_DICTIONARY["_t"]


def test_classify_with_thinking_turns_it_on():
    client = SimpleNamespace(messages=FakeMessages())
    llm.classify(client, "sys", "user", thinking=True, max_tokens=4000)
    call = client.messages.calls[0]
    assert call["thinking"] == {"type": "adaptive"} and call["max_tokens"] == 4000


# ---------- prompt version 2 (routing rules) ----------

def test_original_prompts_unchanged_by_rules_option():
    # Results already recorded under this version must stay comparable.
    zero = prompts.build_system_prompt()
    assert prompts.prompt_version("zero_shot", zero) == "zero_shot-639475c4"
    assert prompts.ROUTING_RULES not in zero


def test_rules_prompt_adds_rules_before_examples():
    ex = [{"label": "mortgage", "text": "My escrow went up."}]
    system = prompts.build_system_prompt(ex, rules=True)
    assert prompts.ROUTING_RULES in system
    assert system.index("How to choose") < system.index("Examples:") < system.index("Reply with JSON")


def test_eval_script_builds_all_variants(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_llm_eval

    examples = tmp_path / "ex.csv"
    pd.DataFrame([{"complaint_id": 1, "label": "mortgage", "text": "Escrow."}]).to_csv(examples, index=False)
    variants = run_llm_eval.build_variants(run_llm_eval.VARIANTS, examples)
    assert "Examples:" not in variants["zero_shot"]["system"]
    assert "How to choose" in variants["zero_shot_rules"]["system"]
    assert "Examples:" in variants["few_shot_rules"]["system"]
    assert len({v["version"] for v in variants.values()}) == 4


def test_pick_examples_saves_choices_and_skips(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import pick_examples

    cands = tmp_path / "cands.csv"
    pd.DataFrame({
        "complaint_id": [1, 2, 3, 4],
        "label": ["mortgage", "mortgage", "student_loan", "student_loan"],
        "text": ["escrow one", "escrow two", "loan one", "loan two"],
    }).to_csv(cands, index=False)
    out = tmp_path / "curated.csv"
    answers = iter(["9", "2", "s"])  # invalid, then pick #2 for mortgage, skip student_loan
    pick_examples.main(["--candidates", str(cands), "--out", str(out)], input_fn=lambda _: next(answers))
    saved = pd.read_csv(out)
    assert saved.to_dict("records") == [{"complaint_id": 2, "label": "mortgage", "text": "escrow two"}]
