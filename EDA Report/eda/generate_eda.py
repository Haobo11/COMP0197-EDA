from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "archive-2" / "historic_demand_2009_2024.csv"
CLEAN_PATH = ROOT / "archive-2" / "historic_demand_2009_2024_noNaN.csv"
OUTPUT_DIR = ROOT / "eda" / "outputs"
FIGURE_DIR = ROOT / "eda" / "figures"


plt.style.use("ggplot")
COLORS = {
    "nd": "#0f4c5c",
    "tsd": "#e36414",
    "wind": "#2a9d8f",
    "solar": "#e9c46a",
    "holiday": "#c1121f",
    "non_holiday": "#003049",
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RAW_PATH)
    clean = pd.read_csv(CLEAN_PATH, parse_dates=["settlement_date"])
    clean["date"] = clean["settlement_date"].dt.date
    clean["year"] = clean["settlement_date"].dt.year
    clean["month"] = clean["settlement_date"].dt.month
    clean["hour"] = (clean["settlement_period"] - 1) / 2
    clean["season"] = clean["month"].map(
        {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Autumn",
            10: "Autumn",
            11: "Autumn",
        }
    )
    return raw, clean


def plot_missingness(raw: pd.DataFrame) -> pd.DataFrame:
    missing = (
        raw.isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_ratio")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    missing = missing[missing["feature"] != ""]
    focus = missing[missing["missing_ratio"] > 0].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(focus["feature"], focus["missing_ratio"] * 100, color="#8d99ae")
    ax.set_ylabel("Missing values (%)")
    ax.set_title("Raw dataset: structural missingness is concentrated in late-added interconnector features")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_01_missingness_full.png", dpi=200)
    plt.close(fig)

    return missing


def plot_monthly_trend(clean: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        clean.set_index("settlement_date")[["nd", "tsd"]]
        .resample("MS")
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["settlement_date"], monthly["nd"], label="ND", color=COLORS["nd"], linewidth=2)
    ax.plot(monthly["settlement_date"], monthly["tsd"], label="TSD", color=COLORS["tsd"], linewidth=2)
    ax.set_title("Monthly average demand shows trend shifts and strong medium-term seasonality")
    ax.set_ylabel("Demand (MW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_02_target_trend.png", dpi=200)
    plt.close(fig)

    return monthly


def plot_intraday_seasonality(clean: pd.DataFrame) -> pd.DataFrame:
    profile = (
        clean.groupby(["season", "hour"])["nd"]
        .mean()
        .reset_index()
        .pivot(index="hour", columns="season", values="nd")
        .reindex(columns=["Winter", "Spring", "Summer", "Autumn"])
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = {
        "Winter": "#264653",
        "Spring": "#2a9d8f",
        "Summer": "#e9c46a",
        "Autumn": "#f4a261",
    }
    for season in profile.columns:
        ax.plot(profile.index, profile[season], label=season, linewidth=2, color=palette[season])
    ax.set_title("Intraday demand profile depends heavily on season")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average ND (MW)")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_03_intraday_seasonality.png", dpi=200)
    plt.close(fig)

    return profile.reset_index()


def plot_yearly_distribution(clean: pd.DataFrame) -> pd.DataFrame:
    yearly_mean = clean.groupby("year")["nd"].mean().rename("mean_nd").reset_index()
    yearly_box = [clean.loc[clean["year"] == year, "nd"].values for year in sorted(clean["year"].unique())]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(
        yearly_box,
        tick_labels=[str(year) for year in sorted(clean["year"].unique())],
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 1.5},
        boxprops={"facecolor": "#bde0fe", "edgecolor": "#457b9d"},
        whiskerprops={"color": "#457b9d"},
        capprops={"color": "#457b9d"},
        flierprops={"marker": ".", "markersize": 1.5, "markerfacecolor": "#e63946", "markeredgecolor": "#e63946"},
    )
    ax.set_title("Demand distribution shifts downward over time, especially after 2020")
    ax.set_xlabel("Year")
    ax.set_ylabel("ND (MW)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_04_yearly_distribution.png", dpi=200)
    plt.close(fig)

    return yearly_mean


def plot_holiday_effect(clean: pd.DataFrame) -> pd.DataFrame:
    holiday_profile = (
        clean.groupby(["is_holiday", "hour"])["nd"]
        .mean()
        .reset_index()
        .pivot(index="hour", columns="is_holiday", values="nd")
        .rename(columns={0: "Non-holiday", 1: "Holiday"})
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        holiday_profile.index,
        holiday_profile["Non-holiday"],
        label="Non-holiday",
        linewidth=2,
        color=COLORS["non_holiday"],
    )
    ax.plot(
        holiday_profile.index,
        holiday_profile["Holiday"],
        label="Holiday",
        linewidth=2,
        color=COLORS["holiday"],
    )
    ax.set_title("Holidays suppress demand across the whole day, not just the peak")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average ND (MW)")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_05_holiday_effect.png", dpi=200)
    plt.close(fig)

    return holiday_profile.reset_index()


def plot_renewables_transition(clean: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        clean.groupby("year")[
            [
                "nd",
                "embedded_wind_generation",
                "embedded_solar_generation",
                "embedded_wind_capacity",
                "embedded_solar_capacity",
            ]
        ]
        .mean()
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(yearly["year"], yearly["nd"], color=COLORS["nd"], linewidth=2, label="Mean ND")
    ax1.set_ylabel("Mean ND (MW)", color=COLORS["nd"])
    ax1.tick_params(axis="y", labelcolor=COLORS["nd"])

    ax2 = ax1.twinx()
    ax2.plot(
        yearly["year"],
        yearly["embedded_wind_capacity"],
        color=COLORS["wind"],
        linewidth=2,
        label="Wind capacity",
    )
    ax2.plot(
        yearly["year"],
        yearly["embedded_solar_capacity"],
        color=COLORS["solar"],
        linewidth=2,
        label="Solar capacity",
    )
    ax2.set_ylabel("Installed embedded capacity (MW)")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.set_title("Renewable capacity growth changes the operating regime over time")
    ax1.set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_06_renewables_transition.png", dpi=200)
    plt.close(fig)

    return yearly


def write_summary(
    raw_missing: pd.DataFrame,
    monthly: pd.DataFrame,
    intraday: pd.DataFrame,
    yearly: pd.DataFrame,
    holiday: pd.DataFrame,
    renewables: pd.DataFrame,
    clean: pd.DataFrame,
) -> None:
    summary = {
        "clean_rows": int(len(clean)),
        "date_start": str(clean["settlement_date"].min()),
        "date_end": str(clean["settlement_date"].max()),
        "nd_mean": float(clean["nd"].mean()),
        "nd_std": float(clean["nd"].std()),
        "tsd_mean": float(clean["tsd"].mean()),
        "holiday_nd_gap_mw": float(
            holiday.loc[holiday["hour"] == 18.0, "Non-holiday"].iloc[0]
            - holiday.loc[holiday["hour"] == 18.0, "Holiday"].iloc[0]
        ),
        "winter_evening_peak_mw": float(intraday.loc[intraday["hour"] == 18.0, "Winter"].iloc[0]),
        "summer_evening_peak_mw": float(intraday.loc[intraday["hour"] == 18.0, "Summer"].iloc[0]),
        "top_missing_features": raw_missing.head(5).to_dict(orient="records"),
        "year_2009_mean_nd": float(yearly.loc[yearly["year"] == 2009, "mean_nd"].iloc[0]),
        "year_2024_mean_nd": float(yearly.loc[yearly["year"] == 2024, "mean_nd"].iloc[0]),
        "wind_capacity_2009": float(renewables.loc[renewables["year"] == 2009, "embedded_wind_capacity"].iloc[0]),
        "wind_capacity_2024": float(renewables.loc[renewables["year"] == 2024, "embedded_wind_capacity"].iloc[0]),
        "solar_capacity_2009": float(renewables.loc[renewables["year"] == 2009, "embedded_solar_capacity"].iloc[0]),
        "solar_capacity_2024": float(renewables.loc[renewables["year"] == 2024, "embedded_solar_capacity"].iloc[0]),
        "nd_tsd_corr": float(clean[["nd", "tsd"]].corr().iloc[0, 1]),
        "wind_nd_corr": float(clean[["nd", "embedded_wind_generation"]].corr().iloc[0, 1]),
        "solar_nd_corr": float(clean[["nd", "embedded_solar_generation"]].corr().iloc[0, 1]),
    }

    (OUTPUT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    raw_missing.to_csv(OUTPUT_DIR / "missingness_table.csv", index=False)
    monthly.to_csv(OUTPUT_DIR / "monthly_trend.csv", index=False)
    intraday.to_csv(OUTPUT_DIR / "intraday_profile.csv", index=False)
    yearly.to_csv(OUTPUT_DIR / "yearly_mean_nd.csv", index=False)
    holiday.to_csv(OUTPUT_DIR / "holiday_profile.csv", index=False)
    renewables.to_csv(OUTPUT_DIR / "renewables_transition.csv", index=False)


def main() -> None:
    ensure_dirs()
    raw, clean = load_data()
    raw_missing = plot_missingness(raw)
    monthly = plot_monthly_trend(clean)
    intraday = plot_intraday_seasonality(clean)
    yearly = plot_yearly_distribution(clean)
    holiday = plot_holiday_effect(clean)
    renewables = plot_renewables_transition(clean)
    write_summary(raw_missing, monthly, intraday, yearly, holiday, renewables, clean)
    print(f"Saved figures to {FIGURE_DIR}")
    print(f"Saved summary tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
