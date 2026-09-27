"""
Visualize model comparison from Week 3.

Run: python -m src.eda.model_comparison
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


RESULTS_PATH = PROJECT_ROOT / "outputs" / "model_results.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "plots"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)

    df["split"] = df["model"].apply(lambda x: "test" if "test" in x else "train")
    df["model_name"] = df["model"].str.replace(r"\s*\(.*\)", "", regex=True)

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric in enumerate(["rmse", "mae", "r2"]):
        sns.barplot(
            data=df, x="model_name", y=metric, hue="split",
            ax=axes[i], palette="Set2"
        )
        axes[i].set_title(f"{metric.upper()} by Model")
        axes[i].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_model_comparison.png", dpi=100)
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR / '07_model_comparison.png'}")

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
