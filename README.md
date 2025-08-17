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

Optional overrides for default model IDs:

- `OPENAI_MODEL` (default: `gpt-5`)
- `ANTHROPIC_MODEL` (default: `claude-opus-4-1-20250805`)
- `GOOGLE_GEMINI_MODEL` (default: `gemini-2.5-pro`)

Supported `--model` values: `chatgpt`, `claude`, `gemini`.

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
# Exp 1: Causal Ablation (expects MedQA CSV)
python experiments/exp1_causal_ablation.py --data data/processed/medqa/medqa_train.csv --model chatgpt --out results/exp1/exp1_medqa_chatgpt.csv

# Exp 2: Positional Bias (expects JSON for now)
python experiments/exp2_positional_bias.py --data data/processed/medqa/positional_bias.json --model claude --out results/exp2/exp2_positional_bias_claude.json

# Exp 3: Hint Injection (expects MedQA CSV)
python experiments/exp3_hint_injection.py --data data/processed/medqa/train_en.parquet --model gemini --out results/exp3/exp3_medqa_gemini.csv

# Exp 4: Real-World Evaluation (expects AskDocs Parquet/CSV/JSON)
python experiments/exp4_realworld_eval.py --data data/processed/askdocs/train_en.parquet --model gemini --out results/exp4/exp4_askdocs_gemini.csv
```
