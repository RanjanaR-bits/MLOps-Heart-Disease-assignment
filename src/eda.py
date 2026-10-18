"""Exploratory data analysis: writes figures to reports/figures and a text summary.

Usage:  python -m src.eda
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (CATEGORICAL_FEATURES, CLEAN_PATH, FIG_DIR, NUMERIC_FEATURES,
                        REPORT_DIR, TARGET)

LABELS = {0: "No disease", 1: "Disease"}


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    print("saved", FIG_DIR / name)


def class_balance(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET].map(LABELS).value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, f"{v} ({v / len(df):.0%})", ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    _save(fig, "01_class_balance.png")


def missing_values(df):
    miss = df.isna().sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=miss.index, y=miss.values, ax=ax)
    ax.set_title("Missing values per column")
    ax.tick_params(axis="x", rotation=60)
    _save(fig, "02_missing_values.png")


def histograms(df):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=df[TARGET].map(LABELS), kde=True,
                     element="step", ax=ax)
        ax.set_title(f"Distribution of {col}")
    axes.ravel()[-1].axis("off")
    _save(fig, "03_numeric_histograms.png")


def boxplots(df):
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(x=df[TARGET].map(LABELS), y=df[col], ax=ax)
        ax.set_xlabel("")
        ax.set_title(col)
    _save(fig, "04_numeric_boxplots_by_target.png")


def categorical_counts(df):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), CATEGORICAL_FEATURES):
        sns.countplot(data=df, x=col, hue=df[TARGET].map(LABELS), ax=ax)
        ax.set_title(col)
        ax.legend(fontsize=7)
    _save(fig, "05_categorical_counts_by_target.png")


def correlation(df):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation heatmap")
    _save(fig, "06_correlation_heatmap.png")


def main():
    sns.set_theme(style="whitegrid", palette="Set2")
    df = pd.read_csv(CLEAN_PATH)
    class_balance(df)
    missing_values(df)
    histograms(df)
    boxplots(df)
    categorical_counts(df)
    correlation(df)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "eda_summary.txt", "w") as fh:
        fh.write(f"Shape: {df.shape}\n\nMissing values:\n{df.isna().sum()}\n\n")
        fh.write(f"Class balance:\n{df[TARGET].value_counts(normalize=True)}\n\n")
        fh.write(f"Describe:\n{df.describe().T}\n\n")
        fh.write("Correlation with target:\n")
        fh.write(str(df.corr(numeric_only=True)[TARGET].sort_values(ascending=False)))
    print("EDA done")


if __name__ == "__main__":
    main()
