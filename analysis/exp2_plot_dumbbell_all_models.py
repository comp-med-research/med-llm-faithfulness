import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def load_exp2_metrics(csv_path: str) -> pd.DataFrame:
	"""Load the merged metrics CSV for experiment 2.

	Expected columns:
	- model
	- Acc_unbiased
	- Acc_bias_gold
	- Acc_bias_wrong
	Other CI columns may be present but are not required for this plot.
	"""
	if not os.path.isfile(csv_path):
		raise FileNotFoundError(f"CSV not found at: {csv_path}")
	return pd.read_csv(csv_path)


def format_axes(ax: plt.Axes, title: str) -> None:
	"""Apply consistent formatting to a subplot axis."""
	ax.set_title(title, fontsize=14, pad=10)
	ax.set_ylabel("Accuracy", fontsize=12)
	ax.yaxis.set_major_formatter(PercentFormatter(1.0))
	ax.grid(axis="y", linestyle=(0, (1, 3)), alpha=0.3)


def plot_model_panel(ax: plt.Axes, row: pd.Series) -> None:
	"""Plot a single model's dumbbell panel.

	- Blue point: average accuracy (unbiased)
	- Red points: biased accuracies for gold and wrong
	- Two categories on x-axis: Bias to Gold, Bias to Wrong
	"""
	categories: List[str] = ["Bias → Gold", "Bias → Wrong"]
	x_positions = [0, 1]

	# Values
	acc_unbiased = float(row["Acc_unbiased"])  # baseline for both categories
	acc_bias_gold = float(row["Acc_bias_gold"])  # biased to gold
	acc_bias_wrong = float(row["Acc_bias_wrong"])  # biased to wrong

	blue_values = [acc_unbiased, acc_unbiased]
	red_values = [acc_bias_gold, acc_bias_wrong]

	# Draw vertical connectors (dumbbells)
	for x, blue_y, red_y in zip(x_positions, blue_values, red_values):
		low, high = (red_y, blue_y) if red_y <= blue_y else (blue_y, red_y)
		ax.vlines(x=x, ymin=low, ymax=high, color="#c7c7c7", linewidth=10, zorder=1)

	# Scatter points
	ax.scatter(x_positions, blue_values, s=80, color="#1f77b4", label="Baseline Accuracy", zorder=2)
	ax.scatter(x_positions, red_values, s=80, color="#d62728", label="Biased Accuracy", zorder=3)

	# Delta annotations: left label centered below, right label centered above
	for x, blue_y, red_y in zip(x_positions, blue_values, red_values):
		delta = (red_y - blue_y) * 100.0
		if x == 0:
			# Place just below the lower point
			y = min(blue_y, red_y) - 0.012
			va = "top"
		else:
			# Place just above the higher point
			y = max(blue_y, red_y) + 0.012
			va = "bottom"
		ax.text(
			x,
			y,
			f"{delta:+.1f}",
			fontsize=11,
			fontweight="bold",
			ha="center",
			va=va,
			color="black",
		)

	# Axes setup
	ax.set_xticks(x_positions)
	ax.set_xticklabels(categories, fontsize=10)
	ax.set_xlim(-0.5, 1.5)


def make_figure(df: pd.DataFrame, out_png: str, out_pdf: str) -> None:
	models = ["Claude 4.1 Opus", "ChatGPT-5", "Gemini Pro 2.5"]
	# Normalize legacy model labels in merged CSVs
	df = df.copy()
	if "model" in df.columns:
		df["model"] = df["model"].replace({
			"Claude": "Claude 4.1 Opus",
			"ChatGPT": "ChatGPT-5",
			"Gemini": "Gemini Pro 2.5",
		})
	df = df.set_index("model").reindex(models).dropna(how="all").reset_index()

	# Determine global y-limits for consistent scaling across panels
	all_values = pd.concat(
		[
			df[["Acc_unbiased", "Acc_bias_gold", "Acc_bias_wrong"]].stack(),
		]
	).astype(float)
	min_y = max(0.0, all_values.min() - 0.05)
	max_y = min(1.0, all_values.max() + 0.05)

	# Smaller width and tighter spacing to enable side-by-side placement in LaTeX
	fig, axes = plt.subplots(1, 3, figsize=(6.2, 3.3), sharey=True)
	for ax, (_, row) in zip(axes, df.iterrows()):
		plot_model_panel(ax, row)
		format_axes(ax, title=str(row["model"]))
		ax.set_ylim(min_y, max_y)

	# Remove y-axis label text from the middle and right panels; keep only on the left
	for ax in axes[1:]:
		ax.set_ylabel("")

	# Move the legend inside the first panel to prevent suptitle overlap
	handles, labels = axes[0].get_legend_handles_labels()
	axes0 = axes[0]
	axes0.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False, fontsize=10)

	# Title and layout: raise title further and reduce inter-panel space
	fig.suptitle("Experiment 2: Baseline vs Biased Accuracy", fontsize=15, y=1.05)
	fig.subplots_adjust(left=0.05, right=0.995, bottom=0.11, top=0.88, wspace=0.01)

	# Ensure output directory exists
	for out_path in [out_png, out_pdf]:
		os.makedirs(os.path.dirname(out_path), exist_ok=True)

	fig.savefig(out_png, dpi=300, bbox_inches="tight")
	fig.savefig(out_pdf, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	csv_path = "/home/gcp_dev/med-llm-faithfulness/results/final/exp2/combined/exp2_metrics_merged.csv"
	out_dir = "/home/gcp_dev/med-llm-faithfulness/results/final/exp2/combined"
	out_png = os.path.join(out_dir, "exp2_dumbbell_all_models.png")
	out_pdf = os.path.join(out_dir, "exp2_dumbbell_all_models.pdf")

	df = load_exp2_metrics(csv_path)
	make_figure(df, out_png, out_pdf)
	print(f"Saved: {out_png}\nSaved: {out_pdf}")


if __name__ == "__main__":
	main()


