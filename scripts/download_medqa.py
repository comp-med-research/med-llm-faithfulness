"""
Download and normalize the MedQA dataset into the project layout.

Raw CSVs will be saved under `data/raw/medqa/medqa_<split>.csv`.
Processed CSVs (schema: id,question,context,options,answer) under
`data/processed/medqa/medqa_<split>.csv`.

Usage:
  python scripts/download_medqa.py --splits train validation test
  python scripts/download_medqa.py --splits train --max-rows 1000
"""

from __future__ import annotations

from datasets import load_dataset

dataset = load_dataset("med_qa_us")
print(dataset)