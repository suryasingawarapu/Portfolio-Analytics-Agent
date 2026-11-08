"""PortfolioAgent: routes a natural-language question to the right tool and runs it.

Two tools are exposed to Gemini via native function calling:
  - query_database(question): anything answerable by SQL over the portfolio database
  - get_sector_exposure(portfolio_name): sector exposure breakdown for one portfolio

A cheap, deterministic check runs first and costs nothing: if the question clearly asks
for sector exposure, the portfolio name is pulled out of the question by word overlap and
the exposure tool runs directly, no LLM call needed for routing at all. Anything else goes
to Gemini's function calling, which decides the tool and extracts its arguments.
"""
import difflib
import logging
import re
import sqlite3
import threading
from pathlib import Path

from google.genai import types

from config import friendly_error, generate_content
from exposure import calculate_exposure, PortfolioNotFound, render_query as render_exposure_query
from sql_tool import answer_sql_question

logger = logging.getLogger(__name__)

DB_FILE = Path(__file__).parent / "portfolio.db"
_EXPOSURE_KEYWORDS = ("sector exposure", "exposure breakdown", "sector breakdown")
_NAME_STOPWORDS = {"fund", "portfolio", "the", "a", "an", "of", "for"}


def _all_portfolio_names() -> list[str]:
    connection = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
    try:
        return [row[0] for row in connection.execute("SELECT portfolio_name FROM portfolios")]
    finally:
        connection.close()


def resolve_portfolio_name(name: str, names: list[str] | None = None) -> str:
    """Resolve a possibly-paraphrased portfolio name to the exact one in the database.

    Exact match first; otherwise a fuzzy match tolerant of wording differences, but only
    when it resolves to exactly one candidate -- ambiguous or no-match cases raise rather
    than silently guessing.
    """
    names = names if names is not None else _all_portfolio_names()
    if name in names:
        return name

    close = difflib.get_close_matches(name, names, n=3, cutoff=0.4)
    if len(close) == 1:
        return close[0]
    if not close:
        raise PortfolioNotFound(f"No portfolio matches '{name}'. Known portfolios: {names}")
    raise PortfolioNotFound(f"'{name}' is ambiguous -- could be any of: {close}")


def _match_name_by_words(question: str, names: list[str]) -> str | None:
    """Pick the portfolio name whose significant words overlap most with the question.

    Handles paraphrases like "international equity" -> "International Equity Fund"
    without any LLM call: every real portfolio name's distinctive words either show up
    in the question or they don't.
    """
    question_words = set(re.findall(r"[a-z]+", question.lower()))
    best_name, best_score = None, 0.0
    for name in names:
        name_words = set(re.findall(r"[a-z]+", name.lower())) - _NAME_STOPWORDS
        if not name_words:
            continue
        score = len(name_words & question_words) / len(name_words)
        if score > best_score:
            best_name, best_score = name, score
    return best_name if best_score >= 0.5 else None


# --- Tool implementations. Docstrings + type hints become the schema Gemini sees. ---

def query_database(question: str) -> dict:
    """Answer a question about portfolios, securities, holdings, transactions, prices,
    performance, or risk metrics by querying the database directly.

    Args:
        question: the user's natural-language question, verbatim.
    """
    return answer_sql_question(question)  # {"sql": ..., "columns": ..., "rows": ...}


def get_sector_exposure(portfolio_name: str) -> dict:
    """Calculate the sector exposure breakdown (percentage of equity value per sector,
    bonds excluded) for one named portfolio.

    Args:
        portfolio_name: the portfolio's name, as mentioned by the user (can be a
            paraphrase -- it will be resolved to the closest real portfolio name).
    """
    resolved = resolve_portfolio_name(portfolio_name)
    return {
        "portfolio_name": resolved,
        "sector_exposure": calculate_exposure(resolved),
        "sql": render_exposure_query(resolved),
    }


_TOOLS = [query_database, get_sector_exposure]
_MAX_TOOL_CALLS = 4  # a safety ceiling, not a target -- most questions need 1, hybrids need 2

# Without this, the model will sometimes guess a parameter (e.g. "which portfolio has the
# highest AUM") instead of looking it up -- a hybrid question needs it to treat an unknown
# fact as a reason to call query_database first, not a reason to assume.
_SYSTEM_INSTRUCTION = (
    "Most questions need exactly ONE tool call -- call it, then stop. Only call a second "
    "tool if the question cannot be fully answered without it (e.g. it explicitly asks for "
    "a sector exposure breakdown of a portfolio that is only identified indirectly, like "
    "'the portfolio with the highest AUM' -- that needs query_database first to find the "
    "name, then get_sector_exposure). If a question refers to a portfolio indirectly like "
    "that, you do not know its name yet: call query_database to find the exact "
    "portfolio_name before using it as an argument -- never guess, assume, or invent one. "
    "Do not call get_sector_exposure unless the question itself asks for a sector exposure "
    "or sector breakdown -- a question about diversification, holdings, sectors counts, or "
    "any other metric is answered by query_database alone."
)

# mode=ANY (forcing an immediate function call) turns out to suppress system_instruction
# entirely -- confirmed empirically: with ANY, the model guessed a portfolio name instead
# of looking it up; with AUTO, it correctly called query_database first. AUTO costs
# slightly more risk of the model answering in plain text without calling a tool at all,
# but that's the right failure mode here (surfaced as an error below) rather than silently
# answering past the system's own guidance.
_AUTO_TOOL_CONFIG = types.GenerateContentConfig(
    system_instruction=_SYSTEM_INSTRUCTION,
    tools=_TOOLS,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)
    ),
)


class PortfolioAgent:
    def __init__(self):
        self.tools_by_name = {tool.__name__: tool for tool in _TOOLS}
        self._cache: dict[str, dict] = {}
        self._cache_lock = threading.Lock()

    def answer_question(self, question: str) -> dict:
        """Same question asked twice returns the saved answer instantly, with no new API
        calls -- in memory only, for this process's lifetime (never written to disk).
        Errors are deliberately never cached: a transient failure (a rate limit, a network
        blip) shouldn't follow a question around after whatever caused it has cleared up.
        """
        key = question.strip().lower()
        with self._cache_lock:
            cached = self._cache.get(key)
        if cached is not None:
            logger.info("Cache hit: %s", question)
            return cached

        result = self._answer_question_uncached(question)
        if not result["error"]:
            with self._cache_lock:
                self._cache[key] = result
        return result

    def _answer_question_uncached(self, question: str) -> dict:
        """Route the question, possibly through several tool calls in sequence (a hybrid
        question needing both tools), and return the LAST step's {tool, args, result,
        error} plus a "steps" list of every call made -- a single-tool question has one
        entry in "steps", identical to the top-level fields, so existing callers that only
        look at the top level are unaffected.

        Never raises: a failure anywhere is caught and returned as a plain-language error
        in the same shape as a successful result.
        """
        logger.info("Question: %s", question)
        names = _all_portfolio_names()
        if any(keyword in question.lower() for keyword in _EXPOSURE_KEYWORDS):
            matched_name = _match_name_by_words(question, names)
            if matched_name:
                logger.info("Routed via keyword fast-path (no LLM call) -> portfolio=%s", matched_name)
                step = self._call_tool("get_sector_exposure", {"portfolio_name": matched_name})
                return {**step, "steps": [step]}

        try:
            return self._route_with_gemini(question)
        except Exception as e:
            logger.error("Routing failed: %s", e)
            error_step = {"tool": None, "args": None, "result": None, "error": friendly_error(e)}
            return {**error_step, "steps": []}

    def _route_with_gemini(self, question: str) -> dict:
        contents = [types.Content(role="user", parts=[types.Part(text=question)])]
        steps = []

        for call_index in range(_MAX_TOOL_CALLS):
            response = generate_content(contents, config=_AUTO_TOOL_CONFIG)
            if not response.function_calls:
                logger.info("Model stopped calling tools after %d step(s)", len(steps))
                if not steps:
                    # Nothing needed looking up (a greeting, small talk, "what can you
                    # do") -- use the model's own reply directly instead of treating "no
                    # tool call" as an error.
                    conversational = {"text": response.text}
                    return {"tool": None, "args": None, "result": conversational, "error": None, "steps": []}
                break  # the model has what it needs after one or more tool calls

            call = response.function_calls[0]
            logger.info("Step %d: Gemini chose tool=%s args=%s", call_index + 1, call.name, dict(call.args))
            step = self._call_tool(call.name, dict(call.args))
            if step["error"]:
                logger.warning("Tool %s failed: %s", call.name, step["error"])
            steps.append(step)

            # Feed the call and its result back so the model can decide on a next step.
            contents.append(response.candidates[0].content)
            response_payload = {"error": step["error"]} if step["error"] else {"result": step["result"]}
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_function_response(name=call.name, response=response_payload)],
            ))

        return {**steps[-1], "steps": steps}

    def _call_tool(self, name: str, args: dict) -> dict:
        tool = self.tools_by_name[name]
        try:
            result = tool(**args)
            return {"tool": name, "args": args, "result": result, "error": None}
        except Exception as e:
            return {"tool": name, "args": args, "result": None, "error": str(e)}


if __name__ == "__main__":
    import sys

    agent = PortfolioAgent()
    question = sys.argv[1] if len(sys.argv) > 1 else "How many portfolios do we have?"
    print(agent.answer_question(question))
