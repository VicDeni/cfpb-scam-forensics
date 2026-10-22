"""Sampling helpers for the evaluation set and few-shot examples (pandas)."""

import pandas as pd

from .text import truncate


def stratified_sample(df: pd.DataFrame, label_col: str, per_class: int, seed: int = 42) -> pd.DataFrame:
    """Up to `per_class` random rows from each class.

    Credit reporting is most of the CFPB data, so a purely random sample would
    barely test the small classes. Capping per class makes macro-F1 meaningful.
    """
    if df.empty:
        return df.reset_index(drop=True)
    parts = [
        group.sample(n=min(per_class, len(group)), random_state=seed)
        for _, group in df.groupby(label_col, sort=True)
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def pick_examples(df: pd.DataFrame, label_col: str, text_col: str, per_class: int = 1,
                  min_chars: int = 200, max_chars: int = 600, seed: int = 42) -> list:
    """Short, typical examples per class for few-shot prompts.

    Prefers narratives between min_chars and max_chars; a class with none in that
    range uses any of its narratives, cut to max_chars. Draw these from the TRAIN
    split only, so no test complaint leaks into a prompt.
    """
    lengths = df[text_col].str.len()
    in_range = (lengths >= min_chars) & (lengths <= max_chars)
    pools = [g[in_range[g.index]] if in_range[g.index].any() else g
             for _, g in df.groupby(label_col, sort=True)]
    if not pools:
        return []
    sample = stratified_sample(pd.concat(pools), label_col, per_class, seed)
    return [
        {"complaint_id": row.get("complaint_id"), "label": row[label_col],
         "text": truncate(row[text_col], max_chars)}
        for _, row in sample.sort_values(label_col).iterrows()
    ]
