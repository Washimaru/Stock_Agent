import asyncio
import json
import re
from datetime import date
from pathlib import Path

import yfinance as yf

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

HAIKU  = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

usage_log: list[dict] = []


def log_usage(agent: str, model: str, usage: dict) -> None:
    usage_log.append({
        "agent":                         agent,
        "model":                         model,
        "input_tokens":                  usage.get("input_tokens", 0) or 0,
        "output_tokens":                 usage.get("output_tokens", 0) or 0,
        "cache_creation_input_tokens":   usage.get("cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens":       usage.get("cache_read_input_tokens", 0) or 0,
    })


def _extract_json(text: str) -> dict:
    """Parse JSON, tolerating optional markdown fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    fenced = re.sub(r"\n?```$", "", fenced).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError(f"No JSON found in agent output:\n{text[:300]}")


async def _run_agent(agent_name: str, model: str, system: str, prompt: str) -> dict:
    """Single-turn Claude agent via Agent SDK.
    Target max_tokens (not enforced by SDK): data=150, debate=600, orchestrator=1000.
    """
    text_parts: list[str] = []
    total_usage: dict = {}

    # Break on ResultMessage to avoid the post-completion CLI exit event.
    async for msg in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            model=model,
            system_prompt=system,
            allowed_tools=[],
            max_turns=3,   # model may use internal think tool before responding
        ),
    ):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
            if msg.usage:
                for k, v in msg.usage.items():
                    if isinstance(v, (int, float)):
                        total_usage[k] = total_usage.get(k, 0) + (v or 0)
        elif isinstance(msg, ResultMessage):
            break

    if total_usage:
        log_usage(agent_name, model, total_usage)

    return _extract_json("".join(text_parts))


# ── yfinance helpers (synchronous I/O offloaded to thread executor) ───────

async def _yf_info(ticker: str) -> dict:
    return await asyncio.to_thread(lambda: yf.Ticker(ticker).info)

async def _yf_news(ticker: str) -> list[dict]:
    return await asyncio.to_thread(lambda: yf.Ticker(ticker).news)

async def _yf_history(ticker: str, period: str = "1mo"):
    return await asyncio.to_thread(lambda: yf.Ticker(ticker).history(period=period))


def _map_analyst_rating(key: str | None) -> str:
    if not key:
        return "Hold"
    k = key.lower().replace("_", "")
    if k in ("buy", "strongbuy"):
        return "Buy"
    if k in ("sell", "strongsell", "underperform"):
        return "Sell"
    return "Hold"


def _extract_news_titles(raw_news: list[dict]) -> list[str]:
    """Handle both yfinance 1.2+ nested (content.title) and older flat (title) shapes."""
    titles = []
    for item in raw_news:
        title = (
            item.get("title")
            or (item.get("content") or {}).get("title")
        )
        if title:
            titles.append(title)
    return titles


# ── Data agents ───────────────────────────────────────────────────────────

async def news_agent(ticker: str) -> dict:
    """Pure Python: fetch real headlines from yfinance — no Claude call."""
    raw = await _yf_news(ticker)
    headlines = _extract_news_titles(raw)[:3]
    return {"headlines": headlines}


async def financials_agent(ticker: str) -> dict:
    """Pure Python: fetch financials from yfinance — no Claude call. File-cached daily.

    yfinance field notes:
      revenueGrowth  — decimal (0.732 = 73.2%); multiply by 100 for pct
      debtToEquity   — percentage-scaled (102.63 ≈ 1.03 ratio); divide by 100
    """
    today = date.today().isoformat()
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{ticker}_{today}.json"

    if cache_file.exists():
        print(f"  [financials_agent] file cache hit: {cache_file}")
        return json.loads(cache_file.read_text())

    info = await _yf_info(ticker)

    pe          = info.get("trailingPE") or info.get("forwardPE")
    rev_growth  = info.get("revenueGrowth")   # decimal
    dte         = info.get("debtToEquity")     # pct-scaled

    data = {
        "pe_ratio":           round(float(pe), 2)               if pe          is not None else None,
        "revenue_growth_pct": round(float(rev_growth) * 100, 2) if rev_growth  is not None else None,
        "debt_to_equity":     round(float(dte) / 100, 2)        if dte         is not None else None,
        "analyst_rating":     _map_analyst_rating(info.get("recommendationKey")),
        "market_cap_b":       round(float(info.get("marketCap", 0)) / 1e9, 2),
    }

    cache_file.write_text(json.dumps(data))
    return data


async def sentiment_agent(ticker: str) -> dict:
    """Fetches real market context from yfinance, then asks Claude to score sentiment."""
    info, raw_news, hist = await asyncio.gather(
        _yf_info(ticker),
        _yf_news(ticker),
        _yf_history(ticker, "1mo"),
    )

    price_change_pct: float | None = None
    if len(hist) >= 2:
        price_change_pct = round(
            (float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[0]) - 1) * 100, 2
        )

    context = {
        "ticker":               ticker,
        "price_change_1mo_pct": price_change_pct,
        "current_price":        info.get("currentPrice"),
        "52w_high":             info.get("fiftyTwoWeekHigh"),
        "52w_low":              info.get("fiftyTwoWeekLow"),
        "analyst_rating":       _map_analyst_rating(info.get("recommendationKey")),
        "recent_news_titles":   _extract_news_titles(raw_news)[:3],
    }

    return await _run_agent(
        "sentiment_agent",
        HAIKU,
        (
            "You are a market sentiment analyst. Use only the data provided — do NOT use any tools. "
            "Given real stock market data (price change, analyst rating, news), return a JSON object "
            "with fields: score (float -1 to 1), catalysts (array of 2-3 strings derived from the "
            "data), key_risks (array of 2-3 strings), and institutional_sentiment (string). "
            "Return raw JSON only — no markdown, no preamble, no explanation."
        ),
        json.dumps(context),
    )
