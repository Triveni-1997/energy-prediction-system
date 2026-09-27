"""
EDA Part 2: Visualizations of energy consumption patterns.
Saves PNG plots to outputs/plots/
Run: python -m src.eda.visualize
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (saves files)
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.load_data import load_energy_data


OUT_DIR = PROJECT_ROOT / "outputs" / "plots"


def ensure_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_daily(energy):
    daily = energy["global_active_power"].resample("D").mean()
    fig, ax = plt.subplots(figsize=(16, 5))
    daily.plot(ax=ax)
    ax.set_title("Daily Average Global Active Power (2006-2010)")
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_daily_consumption.png", dpi=100)
    plt.close(fig)
    print("Saved: 01_daily_consumption.png")


def plot_hourly(energy):
    hourly = energy.groupby(energy.index.hour)["global_active_power"].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    hourly.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Average Energy Consumption by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Power (kW)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_hourly_pattern.png", dpi=100)
    plt.close(fig)
    print("Saved: 02_hourly_pattern.png")


def plot_weekly(energy):
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly = energy.groupby(energy.index.dayofweek)["global_active_power"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(day_names, weekly.values, color="coral")
    ax.set_title("Average Energy Consumption by Day of Week")
    ax.set_ylabel("Average Power (kW)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_weekly_pattern.png", dpi=100)
    plt.close(fig)
    print("Saved: 03_weekly_pattern.png")


def plot_monthly(energy):
    monthly = energy["global_active_power"].resample("ME").mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    monthly.plot(ax=ax, marker="o", color="seagreen")
    ax.set_title("Monthly Average Energy Consumption")
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Month")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_monthly_pattern.png", dpi=100)
    plt.close(fig)
    print("Saved: 04_monthly_pattern.png")


def plot_distribution(energy):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    energy["global_active_power"].hist(bins=50, ax=axes[0], color="steelblue")
    axes[0].set_title("Distribution of Global Active Power")
    axes[0].set_xlabel("Power (kW)")
    energy["global_active_power"].plot(kind="box", ax=axes[1])
    axes[1].set_title("Boxplot of Global Active Power")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_distribution.png", dpi=100)
    plt.close(fig)
    print("Saved: 05_distribution.png")


def plot_correlation(energy):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = energy.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0,
                fmt=".2f", ax=ax, square=True)
    ax.set_title("Correlation Between Energy Variables")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_correlation.png", dpi=100)
    plt.close(fig)
    print("Saved: 06_correlation.png")


def main():
    ensure_dir()
    print("=" * 60)
    print("EDA PART 2: VISUALIZATIONS")
    print("=" * 60)

    energy = load_energy_data()

    print("\nGenerating plots...")
    plot_daily(energy)
    plot_hourly(energy)
    plot_weekly(energy)
    plot_monthly(energy)
    plot_distribution(energy)
    plot_correlation(energy)

    print("\n" + "=" * 60)
    print(f"All plots saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
