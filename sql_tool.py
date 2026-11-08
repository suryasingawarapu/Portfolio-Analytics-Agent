"""Text-to-SQL tool: turns a natural-language question into SQL, validates it, runs it.

The LLM's output is treated as untrusted input. Before anything is executed we check it is
a single read-only SELECT (or WITH ... SELECT) statement, and the query itself runs on a
connection opened in SQLite's read-only mode -- so even a validation gap can't let a
generated query write to the database.
"""
import logging
import re
import sqlite3
from pathlib import Path

from google.genai import types

from config import generate_content
from example_bank import retrieve as retrieve_examples

logger = logging.getLogger(__name__)

DB_FILE = Path(__file__).parent / "portfolio.db"
SCHEMA_FILE = Path(__file__).parent / "database_schema.sql"

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|TRIGGER)\b",
    re.IGNORECASE,
)

_PROMPT_TEMPLATE = """You are a SQLite expert. Given the schema below, write ONE read-only \
SQL query that answers the question. Output ONLY the SQL -- no explanation, no markdown \
code fences.

Match the question's own scope exactly: a question asking for "the total/average/count" \
wants a single aggregated value, not a per-row breakdown. A question asking to "show" or \
"list" several items should include enough identifying columns (e.g. a symbol or name) to \
tell the rows apart, but nothing beyond what the question actually asked for.

Schema:
{schema}
{examples}
Question: {question}

SQL:"""


def _format_examples(examples: list[dict]) -> str:
    if not examples:
        return ""
    blocks = "\n\n".join(f"Q: {e['question']}\nSQL: {e['sql']}" for e in examples)
    return f"\nSimilar solved examples:\n{blocks}\n"

# Deterministic output matters more than creative variety for SQL generation.
_GENERATION_CONFIG = types.GenerateContentConfig(temperature=0)


class UnsafeSQLError(ValueError):
    """Raised when generated SQL isn't a single, read-only SELECT statement."""


def _load_schema() -> str:
    return SCHEMA_FILE.read_text()


def _extract_sql(raw_text: str) -> str:
    """Strip markdown code fences, if the model added them despite being told not to."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip().rstrip(";").strip()


def validate_sql(sql: str) -> None:
    """Raise UnsafeSQLError unless sql is exactly one read-only SELECT/WITH statement."""
    if ";" in sql:
        raise UnsafeSQLError("Multiple statements are not allowed.")
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise UnsafeSQLError("Only SELECT (or WITH ... SELECT) queries are allowed.")
    if _FORBIDDEN_KEYWORDS.search(sql):
        raise UnsafeSQLError("Query contains a write/DDL keyword, which is not allowed.")


def generate_sql(question: str) -> str:
    examples = retrieve_examples(question, k=3)
    logger.debug("Retrieved examples: %s", [e["question"] for e in examples])
    prompt = _PROMPT_TEMPLATE.format(
        schema=_load_schema(), examples=_format_examples(examples), question=question
    )
    response = generate_content(prompt, config=_GENERATION_CONFIG)
    sql = _extract_sql(response.text)
    logger.info("Generated SQL: %s", sql)
    return sql


def run_sql(sql: str) -> tuple[list[str], list[tuple]]:
    """Execute sql on a read-only connection. Returns (column_names, rows)."""
    connection = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
    try:
        cursor = connection.execute(sql)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
    finally:
        connection.close()
    return columns, rows


def answer_sql_question(question: str) -> dict:
    """End-to-end: question -> SQL -> validated -> executed -> result.

    generate_sql() can raise on a genuine API/LLM failure (rate limit, network, etc.) --
    that's left to propagate. Once SQL exists, a safety rejection or an execution error
    (bad column, syntax) is a generation-quality problem, not an infrastructure one, so
    it's captured in the "error" field instead of raised -- callers can tell the two
    failure kinds apart.
    """
    sql = generate_sql(question)
    try:
        validate_sql(sql)
        columns, rows = run_sql(sql)
        logger.info("Executed successfully: %d row(s)", len(rows))
        return {"sql": sql, "columns": columns, "rows": rows, "error": None}
    except (UnsafeSQLError, sqlite3.Error) as e:
        logger.warning("Generated SQL failed: %s", e)
        return {"sql": sql, "columns": None, "rows": None, "error": str(e)}


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "How many portfolios do we have in total?"
    result = answer_sql_question(q)
    print("SQL:", result["sql"])
    print("Columns:", result["columns"])
    for row in result["rows"]:
        print(row)
