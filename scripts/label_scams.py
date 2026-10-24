"""Label the scam sheet one complaint at a time, in the terminal.

    python scripts/label_scams.py

Shows each unlabeled complaint in data/scam_label_sheet.csv and asks for each
field by number, using the same definitions as docs/labeling_guide.md and the
LLM prompt. The labeling rules are applied for you: "not fraud" fills the other
fields with not_applicable, and "unauthorized" skips scam type and contact
channel. The sheet is saved after every complaint, so you can quit any time
and pick up where you left off.

While answering:  u = undo this complaint and start it again
                  s = skip this complaint for now
                  q = save and quit
"""

import argparse
import os
import sys
import textwrap
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from complaint_kit import scam  # noqa: E402

LABEL_COLUMNS = list(scam.FIELDS) + [scam.AMOUNT_FIELD, "notes"]


class Restart(Exception):
    pass


class Skip(Exception):
    pass


class Quit(Exception):
    pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sheet", default=str(ROOT / "data" / "scam_label_sheet.csv"))
    return p.parse_args(argv)


def is_blank(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""


def save(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)  # never leaves a half-written sheet


def command(answer: str) -> None:
    if answer == "u":
        raise Restart
    if answer == "s":
        raise Skip
    if answer == "q":
        raise Quit


def ask_choice(field: str, input_fn) -> str:
    spec = scam.FIELDS[field]
    values = [v for v in spec["values"] if v != scam.NA]
    print(f"\n{field}: {spec['question']}")
    for i, value in enumerate(values, 1):
        desc = textwrap.shorten(spec["values"][value], 70, placeholder="...")
        print(f"  {i:>2}. {value:<26} {desc}")
    while True:
        answer = input_fn(f"{field} (1-{len(values)}): ").strip().lower()
        command(answer)
        if answer.isdigit() and 1 <= int(answer) <= len(values):
            return values[int(answer) - 1]
        print("Type a number from the list (or u / s / q).")


def ask_amount(input_fn) -> str:
    print(f"\n{scam.AMOUNT_FIELD}: total the consumer says they lost, e.g. 1250 or $1,250.00")
    while True:
        answer = input_fn("amount (number, or Enter if none is stated): ").strip()
        command(answer.lower())
        if answer == "" or answer.lower() == "none":
            return "none"
        amount, ok = scam.parse_amount(answer)
        if ok and amount is not None:
            return f"{amount:.2f}"
        print("Type a dollar amount like 1250, or press Enter if no amount is stated.")


def label_one(input_fn) -> dict:
    labels = {"fraud_type": ask_choice("fraud_type", input_fn)}
    if labels["fraud_type"] == "not_fraud":
        labels.update({f: scam.NA for f in scam.FIELDS if f != "fraud_type"})
        labels[scam.AMOUNT_FIELD] = "none"
    else:
        if labels["fraud_type"] == "unauthorized_transaction":
            labels["scam_type"] = labels["contact_channel"] = scam.NA
        else:
            labels["scam_type"] = ask_choice("scam_type", input_fn)
            labels["contact_channel"] = ask_choice("contact_channel", input_fn)
        labels["payment_method"] = ask_choice("payment_method", input_fn)
        labels["refund_status"] = ask_choice("refund_status", input_fn)
        labels[scam.AMOUNT_FIELD] = ask_amount(input_fn)
    note = input_fn("\nnotes (optional, Enter to skip): ").strip()
    command(note.lower() if note.lower() in ("u", "s", "q") else "")
    labels["notes"] = note
    return labels


def main(argv=None, input_fn=input):
    args = parse_args(argv)
    sheet = pd.read_csv(args.sheet, dtype=str, keep_default_na=False)
    for column in LABEL_COLUMNS:
        if column not in sheet:
            sheet[column] = ""
    todo = [i for i in sheet.index if is_blank(sheet.at[i, "fraud_type"])]
    done = len(sheet) - len(todo)
    print(f"{done} of {len(sheet)} complaints labeled; {len(todo)} to go. "
          "Commands at any prompt: u = undo, s = skip, q = save and quit.")

    for i in todo:
        while True:
            print("\n" + "=" * 78)
            print(f"Complaint {done + 1} of {len(sheet)}   (id {sheet.at[i, 'complaint_id']})")
            print("=" * 78)
            print(textwrap.fill(sheet.at[i, "text"], 96))
            try:
                labels = label_one(input_fn)
            except Restart:
                print("\nStarting this complaint again.")
                continue
            except Skip:
                print("\nSkipped. It stays unlabeled for next time.")
                break
            except Quit:
                save(sheet, args.sheet)
                print(f"\nSaved. {done} of {len(sheet)} labeled. Run the script again to continue.")
                return sheet
            for column, value in labels.items():
                sheet.at[i, column] = value
            save(sheet, args.sheet)
            done += 1
            break

    save(sheet, args.sheet)
    remaining = sum(is_blank(v) for v in sheet["fraud_type"])
    print(f"\nDone for now: {len(sheet) - remaining} of {len(sheet)} labeled"
          + (f", {remaining} skipped." if remaining else "."))
    return sheet


if __name__ == "__main__":
    main()
