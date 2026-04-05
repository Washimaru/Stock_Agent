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

- Python 3.9 or higher
- An Anthropic API key — get one free at console.anthropic.com

## Setup
```bash
git clone https://github.com/yourusername/stock-agent.git
cd stock-agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here
```

## Usage
```bash
python main.py
```

You will be prompted to enter a ticker:
