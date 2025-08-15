# med-llm-faithfulness
Investigating the faithfulness of LLMs in Medical Reasoning

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

```
/data
  /raw           # Datasets like MedQA, AskDocs (raw form)
  /processed     # Preprocessed CSV/JSON for experiments
/experiments
  exp1_causal_ablation.py
  exp2_positional_bias.py
  exp3_hint_injection.py
  exp4_realworld_eval.py
/analysis
  metrics.py     # Functions for accuracy, causal density, netflip, etc.
  plotting.py    # Functions for generating tables and figures
  stats.py       # Statistical tests, correlations, CIs
/notebooks
  exp1_analysis.ipynb
  exp2_analysis.ipynb
  exp3_analysis.ipynb
  exp4_analysis.ipynb
/docs
  methodology.md
  results_summary.md
```

## Models and API setup

Set provider API keys via environment variables before running:

- `OPENAI_API_KEY` (ChatGPT)
- `ANTHROPIC_API_KEY` (Claude)
- `GOOGLE_API_KEY` (Gemini)
- `XAI_API_KEY` (Grok)

Optional overrides for default model IDs:

- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `ANTHROPIC_MODEL` (default: `claude-3-5-sonnet-20240620`)
- `GOOGLE_GEMINI_MODEL` (default: `gemini-1.5-pro`)
- `XAI_MODEL` (default: `grok-2-latest`)
- `XAI_BASE_URL` (default: `https://api.x.ai/v1`)

Supported `--model` values: `chatgpt`, `claude`, `gemini`, `grok`.

## Input CSV schema

`data/processed/*.csv` should contain at least:

- `id` (string/int)
- `question` (string)

Optional columns used if present:

- `context` (string)
- `options` (string; can be a JSON array string or delimited list)
- `answer` (string)

## Running an experiment

Examples (choose one provider):

```bash
# ChatGPT (OpenAI)
python experiments/exp1_causal_ablation.py --data data/processed/medqa.csv --model chatgpt --out experiments/outputs/exp1_medqa_chatgpt.csv

# Claude (Anthropic)
python experiments/exp1_causal_ablation.py --data data/processed/medqa.csv --model claude --out experiments/outputs/exp1_medqa_claude.csv

# Gemini (Google)
python experiments/exp1_causal_ablation.py --data data/processed/medqa.csv --model gemini --out experiments/outputs/exp1_medqa_gemini.csv

# Grok (xAI)
python experiments/exp1_causal_ablation.py --data data/processed/medqa.csv --model grok --out experiments/outputs/exp1_medqa_grok.csv
```
