"""Command-line entry point: ask the Portfolio Agent questions interactively.

    python main.py
    python main.py "How many portfolios do we have?"   # one-off, non-interactive
"""
import sys

from agent import PortfolioAgent


def format_result(result: dict) -> str:
    if result["error"]:
        return f"Error: {result['error']}"

    if result["tool"] == "get_sector_exposure":
        data = result["result"]
        lines = [f"Sector exposure for {data['portfolio_name']}:"]
        lines += [f"  {sector}: {pct}%" for sector, pct in data["sector_exposure"].items()]
        return "\n".join(lines)

    if result["tool"] == "query_database":
        data = result["result"]
        lines = [", ".join(data["columns"])]
        lines += [", ".join(str(v) for v in row) for row in data["rows"]]
        return "\n".join(lines) if data["rows"] else "No results."

    if result["tool"] is None:
        # No data lookup was needed -- the model's own conversational reply.
        return result["result"]["text"]

    return str(result)


def main() -> None:
    agent = PortfolioAgent()

    if len(sys.argv) > 1:
        print(format_result(agent.answer_question(" ".join(sys.argv[1:]))))
        return

    print("Portfolio Analytics Agent. Type a question, or 'quit' to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"quit", "exit"}:
            break
        print(format_result(agent.answer_question(question)))


if __name__ == "__main__":
    main()
