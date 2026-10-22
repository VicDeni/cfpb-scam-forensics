import json
import re

import pytest

from complaint_kit import scam

GOOD = {
    "fraud_type": "authorized_scam", "scam_type": "bank_impersonation",
    "contact_channel": "phone_call", "payment_method": "zelle",
    "refund_status": "denied", "amount_lost_usd": 2500,
}


def test_prompt_lists_every_allowed_value_and_rule():
    system = scam.build_system_prompt()
    for field, spec in scam.FIELDS.items():
        for value in spec["values"]:
            assert f"- {value}:" in system
    for rule in scam.RULES:
        assert rule in system
    assert scam.prompt_version(system) == scam.prompt_version(scam.build_system_prompt())


def test_keyword_pattern_works_in_python_and_is_case_insensitive():
    pattern = re.compile(scam.SCAM_KEYWORDS)
    assert pattern.search("I was SCAMMED over Zelle")
    assert pattern.search("they impersonated my bank")
    assert not pattern.search("my overdraft fee was too high")


def test_parse_reply_valid():
    out = scam.parse_reply("Here you go: " + json.dumps(GOOD))
    assert out["fraud_type"] == "authorized_scam" and out["payment_method"] == "zelle"
    assert out["amount_lost_usd"] == 2500.0 and out["amount_ok"]
    assert scam.is_consistent(out)


def test_parse_reply_flags_bad_fields_individually():
    bad = GOOD | {"payment_method": "carrier pigeon", "amount_lost_usd": "a lot"}
    out = scam.parse_reply(json.dumps(bad))
    assert out["payment_method"] == scam.PARSE_ERROR
    assert out["fraud_type"] == "authorized_scam"
    assert out["amount_lost_usd"] is None and not out["amount_ok"]


@pytest.mark.parametrize("reply", [None, "", "no json here", "[1, 2]", "{not json}"])
def test_parse_reply_garbage(reply):
    out = scam.parse_reply(reply)
    assert all(out[f] == scam.PARSE_ERROR for f in scam.FIELDS) and not out["amount_ok"]


@pytest.mark.parametrize("raw, expected", [
    ("$1,250.00", (1250.0, True)), (300, (300.0, True)), (None, (None, True)),
    ("none", (None, True)), ("", (None, True)), (-5, (None, False)), ("lots", (None, False)),
])
def test_parse_amount(raw, expected):
    assert scam.parse_amount(raw) == expected


@pytest.mark.parametrize("value, expected", [
    ("Bank Impersonation", "bank_impersonation"), (" zelle ", None), ("", None), (None, None),
    (float("nan"), None), ("bogus", scam.PARSE_ERROR),
])
def test_normalize_value(value, expected):
    field = "payment_method" if value == " zelle " else "scam_type"
    want = "zelle" if value == " zelle " else expected
    assert scam.normalize_value(field, value) == want


def test_consistency_rules():
    na = scam.NA
    not_fraud = {"fraud_type": "not_fraud", "scam_type": na, "contact_channel": na,
                 "payment_method": na, "refund_status": na, "amount_lost_usd": None}
    assert scam.is_consistent(not_fraud)
    assert not scam.is_consistent(not_fraud | {"payment_method": "zelle"})
    unauth = GOOD | {"fraud_type": "unauthorized_transaction"}
    assert not scam.is_consistent(unauth)
    assert scam.is_consistent(unauth | {"scam_type": na, "contact_channel": na})


def test_wilson_interval():
    low, high = scam.wilson_interval(90, 100)
    assert 0.82 < low < 0.84 and 0.94 < high < 0.95
    assert scam.wilson_interval(0, 0) == (None, None)
    assert scam.wilson_interval(10, 10)[1] == 1.0


def test_amounts_match():
    assert scam.amounts_match(None, None)
    assert scam.amounts_match(1000.0, 1005.0)       # within 1%
    assert not scam.amounts_match(1000.0, 1100.0)
    assert not scam.amounts_match(None, 50.0)


def test_score_counts_skips_blanks_and_reports_typos():
    model = {
        "1": scam.parse_reply(json.dumps(GOOD)),
        "2": scam.parse_reply(json.dumps(GOOD | {"scam_type": "romance", "amount_lost_usd": None})),
    }
    human = [
        {"complaint_id": 1, **GOOD, "amount_lost_usd": "$2,500"},
        {"complaint_id": 2, **GOOD, "contact_channel": "", "refund_status": "denyed",
         "amount_lost_usd": "none"},
        {"complaint_id": 3, **GOOD},  # no model row: ignored
    ]
    results, disagreements, invalid = scam.score(human, model)
    by_field = {r["field"]: r for r in results}
    assert by_field["fraud_type"]["n"] == 2 and by_field["fraud_type"]["accuracy"] == 1.0
    assert by_field["scam_type"]["accuracy"] == 0.5
    assert by_field["contact_channel"]["n"] == 1          # blank cell skipped
    assert by_field["refund_status"]["n"] == 1            # typo not scored...
    assert invalid == [{"complaint_id": "2", "field": "refund_status", "value": "denyed"}]
    assert by_field["amount_lost_usd"]["accuracy"] == 1.0  # "none" matches null
    assert {"complaint_id": "2", "field": "scam_type", "human": "bank_impersonation",
            "model": "romance"} in disagreements


def test_labeling_guide_doc_is_up_to_date():
    from pathlib import Path
    doc = Path(__file__).resolve().parents[1] / "docs" / "labeling_guide.md"
    assert doc.read_text() == scam.labeling_guide_markdown()
