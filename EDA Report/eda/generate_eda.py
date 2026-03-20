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
ANALYSIS_START = pd.Timestamp("2019-01-01 00:00:00")
ANALYSIS_END = pd.Timestamp("2024-12-05 23:30:00")
TARGET = "tsd"
LAG_MAX = 336


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
    clean = clean.loc[
        (clean["settlement_date"] >= ANALYSIS_START)
        & (clean["settlement_date"] <= ANALYSIS_END)
    ].copy()
    clean["date"] = clean["settlement_date"].dt.date
    clean["year"] = clean["settlement_date"].dt.year
    clean["month"] = clean["settlement_date"].dt.month
    clean["day_of_week"] = clean["settlement_date"].dt.dayofweek
    clean["hour"] = (clean["settlement_period"] - 1) / 2
    clean["is_weekend"] = clean["day_of_week"].isin([5, 6]).astype(int)
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
        clean.groupby(["season", "hour"])[TARGET]
        .mean()
        .reset_index()
        .pivot(index="hour", columns="season", values=TARGET)
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
    ax.set_ylabel("Average TSD (MW)")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_03_intraday_seasonality.png", dpi=200)
    plt.close(fig)

    return profile.reset_index()


def plot_yearly_distribution(clean: pd.DataFrame) -> pd.DataFrame:
    yearly_mean = clean.groupby("year")[TARGET].mean().rename("mean_tsd").reset_index()
    yearly_box = [clean.loc[clean["year"] == year, TARGET].values for year in sorted(clean["year"].unique())]

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
    ax.set_ylabel("TSD (MW)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_04_yearly_distribution.png", dpi=200)
    plt.close(fig)

    return yearly_mean


def plot_holiday_effect(clean: pd.DataFrame) -> pd.DataFrame:
    holiday_profile = (
        clean.groupby(["is_holiday", "hour"])[TARGET]
        .mean()
        .reset_index()
        .pivot(index="hour", columns="is_holiday", values=TARGET)
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
    ax.set_ylabel("Average TSD (MW)")
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
                TARGET,
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
    ax1.plot(yearly["year"], yearly[TARGET], color=COLORS["nd"], linewidth=2, label="Mean TSD")
    ax1.set_ylabel("Mean TSD (MW)", color=COLORS["nd"])
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


def plot_weekday_weekend_effect(clean: pd.DataFrame) -> pd.DataFrame:
    profile = (
        clean.groupby(["is_weekend", "hour"])[TARGET]
        .mean()
        .reset_index()
        .pivot(index="hour", columns="is_weekend", values=TARGET)
        .rename(columns={0: "Weekday", 1: "Weekend"})
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(profile.index, profile["Weekday"], label="Weekday", linewidth=2, color="#1d3557")
    ax.plot(profile.index, profile["Weekend"], label="Weekend", linewidth=2, color="#6d597a")
    ax.set_title("Weekday and weekend demand profiles differ most around commuting and business hours")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average TSD (MW)")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_07_weekday_weekend_profile.png", dpi=200)
    plt.close(fig)

    return profile.reset_index()


def plot_feature_correlation(clean: pd.DataFrame) -> pd.DataFrame:
    cols = [
        TARGET,
        "nd",
        "england_wales_demand",
        "embedded_wind_generation",
        "embedded_solar_generation",
        "embedded_wind_capacity",
        "embedded_solar_capacity",
        "pump_storage_pumping",
        "ifa_flow",
        "britned_flow",
        "moyle_flow",
        "east_west_flow",
        "nemo_flow",
        "is_holiday",
        "is_weekend",
    ]
    pearson_corr = clean[cols].corr(method="pearson", numeric_only=True)
    spearman_corr = clean[cols].corr(method="spearman", numeric_only=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, corr, title in [
        (axes[0], pearson_corr, "Pearson Correlation"),
        (axes[1], spearman_corr, "Spearman Correlation"),
    ]:
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle("Feature correlation with target context (2019-2024)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_08_feature_correlation.png", dpi=200)
    plt.close(fig)

    pearson_corr.to_csv(OUTPUT_DIR / "feature_correlation_pearson.csv")
    spearman_corr.to_csv(OUTPUT_DIR / "feature_correlation_spearman.csv")
    return pearson_corr


def plot_lagged_feature_correlation(clean: pd.DataFrame) -> pd.DataFrame:
    features = [
        "nd",
        "england_wales_demand",
        "embedded_wind_generation",
        "embedded_solar_generation",
        "pump_storage_pumping",
        "is_holiday",
        "is_weekend",
    ]
    rows = []
    for feat in features:
        for lag in range(0, LAG_MAX + 1):
            corr = clean[feat].shift(lag).corr(clean[TARGET])
            rows.append({"feature": feat, "lag": lag, "corr": corr})
    lag_corr = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 6))
    for feat in features:
        sub = lag_corr.loc[lag_corr["feature"] == feat]
        ax.plot(sub["lag"], sub["corr"], label=feat, linewidth=1.8)
    for key_lag in [48, 336]:
        ax.axvline(key_lag, color="#555555", linestyle="--", linewidth=1)
    ax.set_title("Lagged correlation: corr(feature[t-k], tsd[t])")
    ax.set_xlabel("Lag k (half-hour steps)")
    ax.set_ylabel("Correlation")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_09_lagged_correlation.png", dpi=200)
    plt.close(fig)

    return lag_corr


def plot_target_autocorrelation(clean: pd.DataFrame) -> pd.DataFrame:
    acf_rows = []
    ts = clean[TARGET]
    for lag in range(1, LAG_MAX + 1):
        acf_rows.append({"lag": lag, "acf": ts.autocorr(lag=lag)})
    acf = pd.DataFrame(acf_rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(acf["lag"], acf["acf"], color="#1d3557", linewidth=2)
    for key_lag in [1, 2, 48, 96, 336]:
        val = acf.loc[acf["lag"] == key_lag, "acf"].iloc[0]
        ax.scatter([key_lag], [val], color="#e63946", s=35)
        ax.text(key_lag, val, f" {key_lag}", fontsize=9)
    ax.set_title("Target autocorrelation of TSD (up to one week)")
    ax.set_xlabel("Lag (half-hour steps)")
    ax.set_ylabel("ACF")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_10_tsd_acf.png", dpi=200)
    plt.close(fig)

    return acf


def plot_seasonal_correlation(clean: pd.DataFrame) -> pd.DataFrame:
    features = [
        "nd",
        "england_wales_demand",
        "embedded_wind_generation",
        "embedded_solar_generation",
        "pump_storage_pumping",
        "is_holiday",
        "is_weekend",
    ]
    rows = []
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        sub = clean.loc[clean["season"] == season]
        for feat in features:
            rows.append(
                {
                    "season": season,
                    "feature": feat,
                    "corr_with_tsd": sub[feat].corr(sub[TARGET]),
                }
            )
    seasonal_corr = pd.DataFrame(rows)
    pivot = seasonal_corr.pivot(index="feature", columns="season", values="corr_with_tsd")

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Season-specific correlation with TSD")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_11_seasonal_correlation.png", dpi=200)
    plt.close(fig)

    return seasonal_corr


def write_summary(
    raw_missing: pd.DataFrame,
    monthly: pd.DataFrame,
    intraday: pd.DataFrame,
    yearly: pd.DataFrame,
    holiday: pd.DataFrame,
    renewables: pd.DataFrame,
    weekday_weekend: pd.DataFrame,
    corr: pd.DataFrame,
    lag_corr: pd.DataFrame,
    target_acf: pd.DataFrame,
    seasonal_corr: pd.DataFrame,
    clean: pd.DataFrame,
) -> None:
    first_year = int(clean["year"].min())
    last_year = int(clean["year"].max())

    summary = {
        "clean_rows": int(len(clean)),
        "date_start": str(clean["settlement_date"].min()),
        "date_end": str(clean["settlement_date"].max()),
        "nd_mean": float(clean["nd"].mean()),
        "nd_std": float(clean["nd"].std()),
        "tsd_mean": float(clean[TARGET].mean()),
        "holiday_tsd_gap_mw": float(
            holiday.loc[holiday["hour"] == 18.0, "Non-holiday"].iloc[0]
            - holiday.loc[holiday["hour"] == 18.0, "Holiday"].iloc[0]
        ),
        "winter_evening_peak_tsd_mw": float(intraday.loc[intraday["hour"] == 18.0, "Winter"].iloc[0]),
        "summer_evening_peak_tsd_mw": float(intraday.loc[intraday["hour"] == 18.0, "Summer"].iloc[0]),
        "top_missing_features": raw_missing.head(5).to_dict(orient="records"),
        "first_year": first_year,
        "last_year": last_year,
        "first_year_mean_tsd": float(yearly.loc[yearly["year"] == first_year, "mean_tsd"].iloc[0]),
        "last_year_mean_tsd": float(yearly.loc[yearly["year"] == last_year, "mean_tsd"].iloc[0]),
        "wind_capacity_first_year": float(
            renewables.loc[renewables["year"] == first_year, "embedded_wind_capacity"].iloc[0]
        ),
        "wind_capacity_last_year": float(
            renewables.loc[renewables["year"] == last_year, "embedded_wind_capacity"].iloc[0]
        ),
        "solar_capacity_first_year": float(
            renewables.loc[renewables["year"] == first_year, "embedded_solar_capacity"].iloc[0]
        ),
        "solar_capacity_last_year": float(
            renewables.loc[renewables["year"] == last_year, "embedded_solar_capacity"].iloc[0]
        ),
        "nd_tsd_corr": float(clean[["nd", TARGET]].corr().iloc[0, 1]),
        "wind_tsd_corr": float(clean[[TARGET, "embedded_wind_generation"]].corr().iloc[0, 1]),
        "solar_tsd_corr": float(clean[[TARGET, "embedded_solar_generation"]].corr().iloc[0, 1]),
        "weekday_weekend_gap_1800_mw": float(
            weekday_weekend.loc[weekday_weekend["hour"] == 18.0, "Weekday"].iloc[0]
            - weekday_weekend.loc[weekday_weekend["hour"] == 18.0, "Weekend"].iloc[0]
        ),
        "top_positive_correlations_with_tsd": corr[TARGET].sort_values(ascending=False).head(6).to_dict(),
        "top_negative_correlations_with_tsd": corr[TARGET].sort_values().head(6).to_dict(),
        "acf_lag_48": float(target_acf.loc[target_acf["lag"] == 48, "acf"].iloc[0]),
        "acf_lag_336": float(target_acf.loc[target_acf["lag"] == 336, "acf"].iloc[0]),
    }

    (OUTPUT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    raw_missing.to_csv(OUTPUT_DIR / "missingness_table.csv", index=False)
    monthly.to_csv(OUTPUT_DIR / "monthly_trend.csv", index=False)
    intraday.to_csv(OUTPUT_DIR / "intraday_profile.csv", index=False)
    yearly.to_csv(OUTPUT_DIR / "yearly_mean_tsd.csv", index=False)
    holiday.to_csv(OUTPUT_DIR / "holiday_profile.csv", index=False)
    renewables.to_csv(OUTPUT_DIR / "renewables_transition.csv", index=False)
    weekday_weekend.to_csv(OUTPUT_DIR / "weekday_weekend_profile.csv", index=False)
    corr.to_csv(OUTPUT_DIR / "feature_correlation.csv")
    lag_corr.to_csv(OUTPUT_DIR / "lag_correlation.csv", index=False)
    target_acf.to_csv(OUTPUT_DIR / "target_autocorrelation.csv", index=False)
    seasonal_corr.to_csv(OUTPUT_DIR / "seasonal_correlation.csv", index=False)

    decisions = pd.DataFrame(
        [
            {
                "eda_signal": "High target autocorrelation at daily/weekly lags",
                "evidence": "figure_10_tsd_acf + target_autocorrelation.csv",
                "modeling_action": "Keep seq_len=48 and consider explicit lag-48/lag-336 features for ablation",
            },
            {
                "eda_signal": "Holiday and weekend profile shifts are systematic",
                "evidence": "figure_05_holiday_effect + figure_07_weekday_weekend_profile",
                "modeling_action": "Retain calendar features (is_holiday, weekend/working-day variants)",
            },
            {
                "eda_signal": "Lagged exogenous correlation differs by feature",
                "evidence": "figure_09_lagged_correlation + lag_correlation.csv",
                "modeling_action": "Prioritize lag engineering for exogenous features with persistent lag correlation",
            },
            {
                "eda_signal": "Feature-target correlation changes by season",
                "evidence": "figure_11_seasonal_correlation + seasonal_correlation.csv",
                "modeling_action": "Check seasonal error slices and keep multivariate inputs for LSTM/Transformer",
            },
            {
                "eda_signal": "Post-2020 level shift with strong annual cycle",
                "evidence": "figure_02_target_trend + figure_04_yearly_distribution",
                "modeling_action": "Use chronological split and report recent-period metrics; keep SARIMA as baseline",
            },
        ]
    )
    decisions.to_csv(OUTPUT_DIR / "eda_to_model_decisions.csv", index=False)


def main() -> None:
    ensure_dirs()
    raw, clean = load_data()
    raw_missing = plot_missingness(raw)
    monthly = plot_monthly_trend(clean)
    intraday = plot_intraday_seasonality(clean)
    yearly = plot_yearly_distribution(clean)
    holiday = plot_holiday_effect(clean)
    renewables = plot_renewables_transition(clean)
    weekday_weekend = plot_weekday_weekend_effect(clean)
    corr = plot_feature_correlation(clean)
    lag_corr = plot_lagged_feature_correlation(clean)
    target_acf = plot_target_autocorrelation(clean)
    seasonal_corr = plot_seasonal_correlation(clean)
    write_summary(
        raw_missing,
        monthly,
        intraday,
        yearly,
        holiday,
        renewables,
        weekday_weekend,
        corr,
        lag_corr,
        target_acf,
        seasonal_corr,
        clean,
    )
    print(f"Saved figures to {FIGURE_DIR}")
    print(f"Saved summary tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
