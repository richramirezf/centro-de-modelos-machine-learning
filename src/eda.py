from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RISK_LABELS = {0: "Good", 1: "Bad"}
GOOD_COLOR = "#2ecc71"
BAD_COLOR = "#e74c3c"


def plot_risk_distribution(df: pd.DataFrame, risk_col: str = "Risk_num", ax=None) -> plt.Axes:
    """Plot the class balance of the target variable Risk (0 = good, 1 = bad)."""
    if ax is None:
        _, ax = plt.subplots()
    counts = df[risk_col].value_counts().sort_index()
    bars = ax.bar(
        [RISK_LABELS.get(value, value) for value in counts.index],
        counts.values,
        color=[GOOD_COLOR, BAD_COLOR],
        edgecolor="black",
    )
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 5, str(int(count)), ha="center", fontweight="bold")
    ax.set_title("Class balance of Risk")
    ax.set_ylabel("Number of clients")
    return ax


def plot_risk_by_account(df: pd.DataFrame, column: str, risk_col: str = "Risk_num", ax=None) -> pd.Series:
    """Compute and plot the % of bad risk grouped by an account status column."""
    if ax is None:
        _, ax = plt.subplots()
    rates = df.groupby(column)[risk_col].mean().mul(100).sort_values(ascending=False)
    bars = ax.bar(rates.index.astype(str), rates.values, color=BAD_COLOR, edgecolor="black")
    for bar, value in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}%", ha="center", fontweight="bold")
    ax.set_title(f"% of Bad Risk by {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("% Bad Risk (defaults)")
    return rates


def main() -> None:
    from src.data_loader import load_german_credit_data
    from src.quality import encode_risk, impute_account_missing

    data_path = Path(__file__).resolve().parent.parent / "data" / "german_credit_data.csv"
    df = encode_risk(impute_account_missing(load_german_credit_data(data_path)))

    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_risk_distribution(df, ax=axes[0])
    plot_risk_by_account(df, "Checking account", ax=axes[1])
    plt.suptitle("German Credit - Risk Exploration")
    plt.tight_layout()
    plt.savefig(reports_dir / "eda_risk.png")
    plt.close()

    print("Saving accounts default rate by status:")
    print(plot_risk_by_account(df, "Saving accounts").round(1).to_string())


if __name__ == "__main__":
    main()