"""Turns a tool's structured result into a natural-language answer.

Kept separate from agent.py on purpose: evaluate.py and the CLI need the structured
result as-is for comparison/display, so this extra Gemini call (one phrasing step on top
of routing + generation) only happens where natural language is actually wanted -- the
chat UI -- not during evaluation.
"""
import time

from config import generate_content, generate_content_stream

_TYPING_DELAY_SECONDS = 0.02

_RESPONSE_PROMPT = """Answer the user's question in 1-3 concise, natural sentences, using \
ONLY the data below. Don't mention SQL, tables, or that a database was queried -- answer \
directly, the way a knowledgeable analyst would.

Question: {question}

Data:
{data}

Answer:"""


def _format_data(result: dict) -> str:
    if result["tool"] == "get_sector_exposure":
        data = result["result"]
        lines = [f"Portfolio: {data['portfolio_name']}"]
        lines += [f"{sector}: {pct}%" for sector, pct in data["sector_exposure"].items()]
        return "\n".join(lines)

    if result["tool"] == "query_database":
        data = result["result"]
        if not data["rows"]:
            return "No rows were returned."
        header = ", ".join(data["columns"])
        rows = "\n".join(", ".join(str(v) for v in row) for row in data["rows"])
        return f"{header}\n{rows}"

    return str(result)


def generate_natural_response(question: str, result: dict) -> str:
    if result["error"]:
        return f"I couldn't answer that: {result['error']}"
    if result["tool"] is None:
        # No data lookup was needed (a greeting, small talk, "what can you do") -- the
        # model already replied conversationally; no need to re-summarize it.
        return result["result"]["text"]
    prompt = _RESPONSE_PROMPT.format(question=question, data=_format_data(result))
    return generate_content(prompt).text.strip()


def _type_out(text: str):
    """Re-chunk text into words with a small delay between each -- a visible typewriter
    effect. Needed because Gemini's own stream arrives as a handful of large chunks sent
    close together, which finishes too fast to read as "streaming" for a short answer."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(_TYPING_DELAY_SECONDS)


def stream_natural_response(question: str, result: dict):
    """Same as generate_natural_response, but yields word-by-word for a visible typewriter
    effect in the UI (e.g. st.write_stream)."""
    if result["error"]:
        yield from _type_out(f"I couldn't answer that: {result['error']}")
        return
    if result["tool"] is None:
        yield from _type_out(result["result"]["text"])
        return
    prompt = _RESPONSE_PROMPT.format(question=question, data=_format_data(result))
    for chunk in generate_content_stream(prompt):
        if chunk.text:
            yield from _type_out(chunk.text)
