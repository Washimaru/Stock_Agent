# Stock_Agent
Given a ticker symbol, this system spins up 5 specialized AI agents:

- **News agent** — fetches and summarizes recent headlines
- **Financials agent** — pulls P/E ratio, revenue growth, analyst rating
- **Sentiment agent** — scores market tone from -10 to +10
- **Bull agent** — makes the strongest buy case
- **Bear agent** — makes the strongest sell case

An orchestrator synthesizes both sides into a final Buy / Hold / Avoid verdict.

## Features
- Parallel agent execution with asyncio
- File-based caching for financials (no repeat API calls)
- Token usage tracker with cost breakdown per agent
- Model routing: Haiku for data tasks, Sonnet for reasoning

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here
```

## Usage
```bash
python main.py
# Enter ticker when prompted: AAPL
```
