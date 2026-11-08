"""Sector exposure calculator.

Given a portfolio, works out what percentage of its EQUITY value sits in each sector.
Bonds are excluded entirely (they have no sector), and the remaining stock weights are
renormalized so the reported percentages sum to 100% of the equity portion -- not of the
whole portfolio, which would understate every sector for any portfolio that also holds bonds.

No LLM involved: this is plain arithmetic over the database, kept separate from the SQL
tool so the numbers are always exactly reproducible.
"""
import sqlite3
from pathlib import Path

DB_FILE = Path(__file__).parent / "portfolio.db"

_QUERY = """
SELECT sec.sector_name, h.quantity * s.current_price AS market_value
FROM holdings h
JOIN securities s ON h.security_id = s.security_id
JOIN portfolios p ON h.portfolio_id = p.portfolio_id
JOIN sectors sec ON s.sector_id = sec.sector_id
WHERE p.portfolio_name = ? AND s.asset_type = 'Stock'
"""


class PortfolioNotFound(ValueError):
    pass


def _portfolio_exists(connection: sqlite3.Connection, portfolio_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM portfolios WHERE portfolio_name = ?", (portfolio_name,)
    ).fetchone()
    return row is not None


def render_query(portfolio_name: str) -> str:
    """The exact SQL used internally, with the parameter filled in -- for DISPLAY only.
    The real query above always runs parameterized; this is just a readable rendering of
    it for showing "what ran" in a UI, never re-executed."""
    return _QUERY.replace("?", f"'{portfolio_name}'").strip()


def calculate_exposure(portfolio_name: str, db_file: Path = DB_FILE) -> dict[str, float]:
    """Return {sector_name: percentage} for the equity holdings of one portfolio.

    Percentages are rounded to 2 decimal places and sum to 100.0 (give or take rounding),
    computed over stocks only -- bonds are excluded before the percentages are taken.
    """
    connection = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    try:
        if not _portfolio_exists(connection, portfolio_name):
            raise PortfolioNotFound(f"No portfolio named '{portfolio_name}'")

        rows = connection.execute(_QUERY, (portfolio_name,)).fetchall()
    finally:
        connection.close()

    if not rows:
        return {}  # portfolio exists but holds no stocks (e.g. a pure bond fund)

    sector_totals: dict[str, float] = {}
    for sector_name, market_value in rows:
        sector_totals[sector_name] = sector_totals.get(sector_name, 0.0) + market_value

    equity_total = sum(sector_totals.values())
    return {
        sector: round(value / equity_total * 100, 2)
        for sector, value in sorted(sector_totals.items(), key=lambda item: -item[1])
    }


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "Tech Innovation Fund"
    for sector, pct in calculate_exposure(name).items():
        print(f"{sector}: {pct}%")
