# Methodology

This document outlines the experimental methodology for assessing LLM faithfulness in medical reasoning across four experiments: causal ablation, positional bias, hint injection, and real-world evaluation.

- Data sources: MedQA, AskDocs, and related clinical QA datasets.
- Preprocessing: Normalize fields into a unified JSON schema with `id`, `question`, `context`, `options`, and `answer`.
- Evaluation metrics: accuracy, causal density, netflip, confidence intervals, and significance tests.
