import asyncio
import json

from agents import SONNET, _run_agent, log_usage


def _trim_context(ticker: str, sentiment: dict, financials: dict) -> dict:
    """Trim to only the 7 fields the bull/bear agents need."""
    return {
        "ticker": ticker,
        "sentiment": {
            "score":     sentiment.get("score"),
            "catalysts": sentiment.get("catalysts"),
            "key_risks": sentiment.get("key_risks"),
        },
        "financials": {
            "pe_ratio":           financials.get("pe_ratio"),
            "revenue_growth_pct": financials.get("revenue_growth_pct"),
            "analyst_rating":     financials.get("analyst_rating"),
        },
    }


async def bull_agent(context: dict) -> dict:
    return await _run_agent(
        "bull_agent",
        SONNET,
        (
            "You are a bullish equity analyst. Use only the data provided — do NOT use any tools. "
            "Construct the strongest possible bull case. Cite the specific metrics provided. "
            "Return a JSON object with fields: thesis (string), key_points (array of 3 strings), "
            "price_target_upside_pct (float). "
            "Return raw JSON only — no markdown, no preamble."
        ),
        json.dumps(context),
    )


async def bear_agent(context: dict) -> dict:
    return await _run_agent(
        "bear_agent",
        SONNET,
        (
            "You are a bearish equity analyst. Use only the data provided — do NOT use any tools. "
            "Construct the strongest possible bear case. Cite the specific metrics provided. "
            "Return a JSON object with fields: thesis (string), key_points (array of 3 strings), "
            "price_target_downside_pct (float). "
            "Return raw JSON only — no markdown, no preamble."
        ),
        json.dumps(context),
    )


async def synthesize_report(
    ticker: str,
    news: dict,
    financials: dict,
    sentiment: dict,
) -> dict:
    trimmed = _trim_context(ticker, sentiment, financials)

    # Bull and bear run in parallel
    bull, bear = await asyncio.gather(
        bull_agent(trimmed),
        bear_agent(trimmed),
    )

    return await _run_agent(
        "orchestrator",
        SONNET,
        (
            "You are a senior portfolio manager synthesizing a final investment research report. "
            "Use only the data provided — do NOT use any tools. "
            "Given bull and bear cases plus supporting data, produce a balanced verdict. "
            "Return a JSON object with fields: verdict (string — Buy, Hold, or Sell), "
            "confidence (float 0-1), summary (string, 2-3 sentences), "
            "bull_case (string), bear_case (string), "
            "key_metrics (object with pe_ratio, revenue_growth_pct, analyst_rating), "
            "recommendation (string). "
            "Return raw JSON only — no markdown, no preamble."
        ),
        json.dumps({
            "ticker":     ticker,
            "news":       news,
            "financials": financials,
            "sentiment":  sentiment,
            "bull_case":  bull,
            "bear_case":  bear,
        }),
    )
