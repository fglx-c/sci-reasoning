# Simple ELO Evaluation System

A simplified ELO tournament evaluation system adapted from CoI-Agent for comparing research ideas using LLM judges.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=""
```

## How It Works

1. **Pairwise Comparisons**: Each idea is compared against every other idea
2. **LLM Judging**: Uses the original CoI-Agent prompt with 5 criteria:
3. **ELO Scoring**: Updates scores based on win/loss/tie results
4. **Ranking**: Final ranking by ELO score

## Configuration Options

- `model`: OpenAI model name (default: "gpt-4")
- `temperature`: LLM temperature (default: 0.7)
- `max_tokens`: Max response tokens (default: 4000)
- `parallel_evaluations`: Enable parallel processing (default: True)
- `max_concurrent_evaluations`: Max parallel requests (default: 5)
- `save_results`: Save results to JSON (default: True)

## Retry Logic

Uses tenacity library for automatic retries:
- **Fixed delay**: 10 seconds between retries
- **Max attempts**: 10 retries per API call
- **Handles**: Rate limits, network issues, temporary API failures

---