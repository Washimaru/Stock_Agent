import asyncio
import json
import sys
from datetime import date
from pathlib import Path

from agents import news_agent, financials_agent, sentiment_agent, usage_log
from orchestrator import synthesize_report

# Pricing per million tokens (input / output / cache-write / cache-read)
PRICING = {
    "claude-haiku-4-5-20251001": {
        "input":       1.00,
        "output":      5.00,
        "cache_write": 1.25,   # 1.25× input
        "cache_read":  0.10,   # 0.10× input
    },
    "claude-sonnet-4-6": {
        "input":       3.00,
        "output":     15.00,
        "cache_write": 3.75,   # 1.25× input
        "cache_read":  0.30,   # 0.10× input
    },
}

W = (22, 24, 7, 7, 8, 7, 11)   # column widths


def _row(*cells) -> str:
    return (
        f"{cells[0]:<{W[0]}} {cells[1]:<{W[1]}} "
        f"{cells[2]:>{W[2]}} {cells[3]:>{W[3]}} "
        f"{cells[4]:>{W[4]}} {cells[5]:>{W[5]}} "
        f"{cells[6]:>{W[6]}}"
    )


def print_cost_table() -> None:
    line = "─" * (sum(W) + len(W) - 1)
    print(f"\n╒{'═' * (sum(W) + len(W) - 1)}╕")
    print(f"│{'  TOKEN USAGE & ESTIMATED COST':^{sum(W) + len(W) - 1}}│")
    print(f"╞{'═' * (sum(W) + len(W) - 1)}╡")
    print("│ " + _row("Agent", "Model", "In", "Out", "CWrite", "CRead", "Cost") + " │")
    print(f"╞{line}╡")

    total = 0.0
    for e in usage_log:
        p = PRICING.get(e["model"], {k: 0 for k in ("input","output","cache_write","cache_read")})
        cost = (
            e["input_tokens"]                  * p["input"]       / 1_000_000
            + e["output_tokens"]               * p["output"]      / 1_000_000
            + e["cache_creation_input_tokens"] * p["cache_write"] / 1_000_000
            + e["cache_read_input_tokens"]     * p["cache_read"]  / 1_000_000
        )
        total += cost
        print("│ " + _row(
            e["agent"],
            e["model"],
            e["input_tokens"],
            e["output_tokens"],
            e["cache_creation_input_tokens"],
            e["cache_read_input_tokens"],
            f"${cost:.5f}",
        ) + " │")

    print(f"╞{line}╡")
    total_label = "TOTAL"
    padding = sum(W[:6]) + len(W[:6]) - 1 - len(total_label)
    print(f"│ {total_label}{' ' * padding} ${total:>{W[6]-1}.5f} │")
    print(f"╘{'═' * (sum(W) + len(W) - 1)}╛")


async def main(ticker: str) -> None:
    banner = f"  Stock Research Agent — {ticker}  "
    print(f"\n{'='*len(banner)}")
    print(banner)
    print(f"{'='*len(banner)}\n")

    print("[1/2] Running data agents in parallel (news · financials · sentiment)…")
    news, financials, sentiment = await asyncio.gather(
        news_agent(ticker),
        financials_agent(ticker),
        sentiment_agent(ticker),
    )

    print("\n── News ─────────────────────────────────")
    print(json.dumps(news, indent=2))
    print("\n── Financials ───────────────────────────")
    print(json.dumps(financials, indent=2))
    print("\n── Sentiment ────────────────────────────")
    print(json.dumps(sentiment, indent=2))

    print("\n[2/2] Running bull/bear debate in parallel, then synthesizing report…")
    report = await synthesize_report(ticker, news, financials, sentiment)

    print(f"\n{'='*50}")
    print(f"  FINAL REPORT: {ticker}")
    print(f"{'='*50}")
    print(json.dumps(report, indent=2))

    print_cost_table()

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_file = reports_dir / f"{ticker}_{date.today().isoformat()}.json"
    report_file.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {report_file}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    asyncio.run(main(ticker))
