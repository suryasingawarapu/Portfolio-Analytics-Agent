"""Evaluator: runs every question in ground_truth_dataset.json through the agent and
checks the answer against the ground truth.

text2sql questions: the agent's generated SQL is executed and its result rows are
compared against the reference SQL's result rows -- not a string diff of the SQL itself,
since two differently-written queries can be equally correct. Row order only matters when
the reference query uses LIMIT (a "top N" question); otherwise rows are compared as a set,
since GROUP BY/JOIN order isn't semantically meaningful unless the question asked for it.

exposure_calculator questions: ground truth only gives the expected tool name and
parameters (no expected numbers), so correctness here means the agent chose the exposure
tool and resolved the same portfolio.
"""
import json
from pathlib import Path

from agent import PortfolioAgent
from sql_tool import run_sql

GROUND_TRUTH_FILE = Path(__file__).parent / "ground_truth_dataset.json"

# Our tool's actual name vs. the name used in ground_truth_dataset.json.
_TOOL_NAME_ALIASES = {"exposure_calculator": "get_sector_exposure"}


def _round_floats(row: tuple, places: int = 2) -> tuple:
    return tuple(round(v, places) if isinstance(v, float) else v for v in row)


def _rows_equivalent(expected: tuple, actual: tuple) -> bool:
    """Two rows are equivalent if one's values are a subset of the other's.

    A free-form "write SQL for this question" task has no single correct column set --
    the agent might reasonably select fewer or more identifying columns than the one
    reference query happened to choose. What matters is that the values it DID return are
    actually correct, not that it picked the exact same columns.
    """
    return set(expected) <= set(actual) or set(actual) <= set(expected)


def _unordered_rows_match(expected_rows: list[tuple], actual_rows: list[tuple]) -> bool:
    if len(expected_rows) != len(actual_rows):
        return False
    remaining = list(actual_rows)
    for expected_row in expected_rows:
        match = next((row for row in remaining if _rows_equivalent(expected_row, row)), None)
        if match is None:
            return False
        remaining.remove(match)
    return True


def _ordered_rows_match(expected_rows: list[tuple], actual_rows: list[tuple]) -> bool:
    return len(expected_rows) == len(actual_rows) and all(
        _rows_equivalent(e, a) for e, a in zip(expected_rows, actual_rows)
    )


# Diagnostic categories -- "where" a question failed, not just that it did. Distinguishing
# these matters because they point at completely different fixes: a wrong tool means the
# router needs work, wrong logic means the SQL/exposure math needs work, and an agent error
# means the LLM call itself failed (rate limit, network, etc.) -- nothing to do with
# correctness at all.
AGENT_ERROR = "agent_error"
WRONG_TOOL = "wrong_tool"
WRONG_LOGIC = "wrong_logic"
PASS = "pass"


def _check_sql_question(question: dict, agent_result: dict) -> tuple[str, str]:
    if agent_result["error"]:
        return AGENT_ERROR, agent_result["error"]
    if agent_result["tool"] != "query_database":
        return WRONG_TOOL, f"chose '{agent_result['tool']}' instead of 'query_database'"
    if agent_result["result"]["error"]:
        # The SQL itself was bad (unsafe or failed to execute) -- a generation mistake,
        # not an infrastructure failure.
        return WRONG_LOGIC, f"generated SQL failed: {agent_result['result']['error']}"

    reference_sql = question["ground_truth"]["sql_query"]
    try:
        _, expected_rows = run_sql(reference_sql)
    except Exception as e:
        return AGENT_ERROR, f"reference SQL itself failed: {e}"

    actual_rows = agent_result["result"]["rows"]  # already executed once by the agent
    expected_rows = [_round_floats(r) for r in expected_rows]
    actual_rows = [_round_floats(r) for r in actual_rows]

    if "LIMIT" in reference_sql.upper():
        matched = _ordered_rows_match(expected_rows, actual_rows)
    else:
        matched = _unordered_rows_match(expected_rows, actual_rows)

    if matched:
        return PASS, ""
    return WRONG_LOGIC, f"expected {expected_rows}, got {actual_rows}"


def _check_exposure_question(question: dict, agent_result: dict) -> tuple[str, str]:
    expected_tool = _TOOL_NAME_ALIASES[question["ground_truth"]["tool_name"]]
    expected_portfolio = question["ground_truth"]["parameters"]["portfolio_name"]

    if agent_result["error"]:
        return AGENT_ERROR, agent_result["error"]
    if agent_result["tool"] != expected_tool:
        return WRONG_TOOL, f"chose '{agent_result['tool']}' instead of '{expected_tool}'"

    actual_portfolio = agent_result["result"]["portfolio_name"]  # the resolved name
    if actual_portfolio == expected_portfolio:
        return PASS, ""
    return WRONG_LOGIC, f"resolved to '{actual_portfolio}', expected '{expected_portfolio}'"


def evaluate() -> list[dict]:
    agent = PortfolioAgent()
    questions = json.loads(GROUND_TRUTH_FILE.read_text())["questions"]
    results = []

    for question in questions:
        agent_result = agent.answer_question(question["question"])
        checker = _check_sql_question if question["type"] == "text2sql" else _check_exposure_question
        category, detail = checker(question, agent_result)

        results.append({
            "id": question["id"],
            "difficulty": question["difficulty"],
            "question": question["question"],
            "passed": category == PASS,
            "category": category,
            "detail": detail,
        })

    return results


def print_report(results: list[dict]) -> None:
    for r in results:
        status = "PASS" if r["passed"] else f"FAIL ({r['category']})"
        print(f"[{status}] #{r['id']} ({r['difficulty']}) {r['question']}")
        if not r["passed"]:
            print(f"       {r['detail']}")

    print("\n--- Accuracy by difficulty ---")
    for difficulty in ("easy", "medium", "hard"):
        tier = [r for r in results if r["difficulty"] == difficulty]
        if not tier:
            continue
        passed = sum(r["passed"] for r in tier)
        print(f"{difficulty:8s} {passed}/{len(tier)}")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\n--- Failures by category ---")
        for category in (AGENT_ERROR, WRONG_TOOL, WRONG_LOGIC):
            count = sum(r["category"] == category for r in failures)
            if count:
                print(f"{category:12s} {count}")

    total_passed = sum(r["passed"] for r in results)
    print(f"\nOverall: {total_passed}/{len(results)}")


if __name__ == "__main__":
    print_report(evaluate())
