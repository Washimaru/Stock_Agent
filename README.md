# Stock_agent
# Stock Research Agent

A command-line tool that researches any stock ticker and produces a 
structured investment report — covering recent news, key financials, 
market sentiment, and a final Buy / Hold / Avoid verdict.

## What it does

Enter a ticker symbol and get back a report that includes:

- Recent headlines and their overall tone
- Key financial metrics (P/E ratio, revenue growth, analyst rating)
- Sentiment score from -10 (very bearish) to +10 (very bullish)
- A bull case — the strongest reasons to buy
- A bear case — the strongest reasons to avoid
- A final verdict synthesizing both sides

## Requirements

- Python 3.10 or higher
- An Anthropic API key — get one free at console.anthropic.com
- `yfinance`
- `claude-agent-sdk`

## Setup
```bash
git clone https://github.com/Washimaru/Stock_Agent.git
cd Stock_Agent
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install anthropic yfinance claude-agent-sdk
export ANTHROPIC_API_KEY=your_key_here
```

## Usage
```bash
python main.py <TICKER>
```

Example:
```bash
python main.py AAPL
```
