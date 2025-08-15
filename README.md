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

Examples (one per experiment):

```bash
# Exp 1: Causal Ablation (structured QA, e.g., MedQA CSV)
python experiments/exp1_causal_ablation.py --data data/processed/medqa/medqa_train.csv --model chatgpt --out results/exp1_medqa_chatgpt.csv

# Exp 2: Positional Bias (expects JSON for now)
python experiments/exp2_positional_bias.py --data data/processed/medqa/positional_bias.json --model claude --out results/exp2_positional_bias_claude.json

# Exp 3: Hint Injection (expects JSON for now)
python experiments/exp3_hint_injection.py --data data/processed/medqa/hint_injection.json --model grok --out results/exp3_hint_injection_grok.json

# Exp 4: Real-World Evaluation (forum posts, e.g., AskDocs Parquet/CSV/JSON)
python experiments/exp4_realworld_eval.py --data data/raw/askdocs/askdocs_train_en.parquet --model gemini --out results/exp4_askdocs_gemini.csv
```
