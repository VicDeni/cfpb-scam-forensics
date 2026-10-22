"""SQL for writing the data dictionary into Unity Catalog as comments."""

from .schema import DATA_DICTIONARY


def _quote(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def comment_statements(table_fqn: str, table_key: str, existing_columns) -> list:
    """COMMENT statements for one table, skipping columns it doesn't have."""
    table_doc, column_docs = DATA_DICTIONARY[table_key]
    statements = [f"COMMENT ON TABLE {table_fqn} IS {_quote(table_doc)}"]
    existing = set(existing_columns)
    for column, doc in column_docs.items():
        if column in existing:
            statements.append(
                f"ALTER TABLE {table_fqn} ALTER COLUMN `{column}` COMMENT {_quote(doc)}"
            )
    return statements
