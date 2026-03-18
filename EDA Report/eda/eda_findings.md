# EDA Findings for TSD Forecasting (2019-2024)

This note summarises the findings that directly support our final setup:

- target variable: `tsd`
- modelling window: `2019-01-01` to `2024-12-05`
- main models: `LSTM` and `Transformer`
- baselines: `CNN` and `ARIMA`
- feature strategy: use all available features in the cleaned dataset

## 1. Why we use 2019-2024

Yes, this can be presented as part of `Problem Solving`, especially under technical hurdles such as data quality and hardware limitations.

We chose the `2019-01-01` to `2024-12-05` window for three practical reasons.

First, it avoids most of the structural missingness problem in earlier years. Several interconnector-related variables are unavailable for long periods before 2019, and those gaps are caused by system evolution rather than random noise. If we keep the earlier years, we either have to drop more variables or design a more complicated masking and imputation pipeline.

Second, it gives us a more consistent operating regime. The recent years better reflect the current relationship between demand, renewable generation, interconnector flows, and calendar effects. Since our goal is forecasting under the current system, recent data is more informative than very old data collected under materially different conditions.

Third, it is more practical for CPU-only training. After filtering, the dataset still contains `103,956` half-hourly rows, which is enough for robust training and evaluation, while reducing training time and iteration cost compared with the full 2009-2024 history.

So this is not just a convenience choice. It is a problem-solving decision that addresses:

- structural missingness in early years
- regime inconsistency across the full historical range
- CPU training constraints for sequence models

## 2. Dataset overview

- Frequency: half-hourly
- Target: `tsd`
- Time span used for modelling: `2019-01-01 00:00:00` to `2024-12-05 23:30:00`
- Number of rows: `103,956`
- TSD mean: `29,035.9 MW`
- TSD range: `16,513 MW` to `48,800 MW`

We use the cleaned dataset rather than the raw table because the raw data contains both impossible records and structural missingness that would otherwise contaminate training.

## 3. Data quality checks

The raw dataset contains a small number of clearly problematic records:

- `32` rows with `settlement_period > 48`
- `479` rows with `tsd = 0`

These values are not physically credible for a half-hourly electricity demand series. If we trained directly on the raw table, sequence models could waste capacity fitting artefacts instead of real dynamics. This is why all experiments start from the cleaned dataset.

## 4. Monthly trend and regime change

![Monthly trend](figures/figure_02_target_trend.png)

The monthly average `tsd` plot shows two clear patterns.

The first is strong annual seasonality. Demand is systematically higher in winter and lower in summer. Monthly averages are highest in January (`34,578 MW`) and December (`33,244 MW`), while the lowest values appear in June (`25,262 MW`) and August (`25,338 MW`).

The second is a changing level across years. Yearly mean `tsd` falls from `30,539 MW` in 2019 to `27,974 MW` in 2024, with a clear drop in 2020 and a lower-demand regime persisting afterward.

This figure affects the later modelling steps in three ways:

- train, validation, and test splits must be chronological
- recent-period evaluation is more important than average performance across all years
- the model must learn both repeated seasonality and slower distribution shift

This is one reason we selected LSTM and Transformer as the main models. They are better suited than simpler fixed-structure models to learning both local temporal patterns and broader context changes.

## 5. Intraday and seasonal structure

![Intraday seasonality](figures/figure_03_intraday_seasonality.png)

This figure shows that the shape of the day is highly regular.

Demand is lowest in the early morning. At `04:00`, average `tsd` is about:

- `25,750 MW` in winter
- `23,151 MW` in spring
- `21,102 MW` in summer
- `22,685 MW` in autumn

Demand then rises sharply after around `06:00`. By `08:00`, the morning ramp is already clear, and the strongest peak appears around `18:00`. At that hour, average `tsd` reaches:

- `41,238 MW` in winter
- `33,312 MW` in spring
- `29,292 MW` in summer
- `36,268 MW` in autumn

This tells us several things:

- the series has strong intraday seasonality
- peak hours are much harder and more important than low-demand hours
- season changes the amplitude of the daily pattern, not just the average level

For the next steps, this supports using sequence models with enough capacity to learn recurring half-hour structure. It also justifies keeping all time-related and system-related features, because a fixed daily template is clearly not enough.

This figure is also useful for baseline interpretation:

- `ARIMA` provides a classical reference for recurring temporal structure
- `CNN` can test whether local temporal filters are enough
- `LSTM` and `Transformer` can then be judged on whether they better capture ramping and peak structure

## 6. Calendar effects

![Holiday effect](figures/figure_05_holiday_effect.png)

Holiday demand is consistently lower than non-holiday demand across the full day. The average `tsd` is `29,107 MW` on non-holidays and `25,955 MW` on holidays. Around `18:00`, the holiday gap is about `3,972 MW`.

This suggests that holidays behave like a systematic downward shift rather than isolated anomalies. In other words, holiday information is not optional decoration; it changes the baseline level of the series.

![Weekday weekend profile](figures/figure_07_weekday_weekend_profile.png)

The weekday-weekend comparison shows another stable calendar pattern. At `18:00`, the weekday-weekend gap is about `3,638 MW`, with the largest differences appearing during business and evening hours.

These two figures together support our decision to keep all available cleaned features and derived calendar features. For modelling, the implication is straightforward:

- holiday and weekday information should be explicit inputs
- they should not be left for the model to infer only from recent history

This is especially important for CPU-only experiments, because giving the model the right signals directly is often more efficient than relying on a larger model to discover them from scratch.

## 7. Renewable and system features

![Renewables transition](figures/figure_06_renewables_transition.png)

The renewable transition figure shows that system conditions continue to evolve even within the 2019-2024 window. Installed solar capacity keeps rising, wind capacity remains high, and the target level changes over time.

This means the same hour in the same season does not always correspond to the same system state. Exogenous variables help explain that difference.

From the correlation analysis, `tsd` is strongly aligned with other demand-related variables such as:

- `nd` (`0.980`)
- `england_wales_demand` (`0.979`)

Other variables appear less dominant individually, but still carry useful system information:

- `pump_storage_pumping` has a moderate negative relationship with `tsd` (`-0.377`)
- `is_weekend` is also clearly negative (`-0.266`)
- renewable generation and capacity variables show weaker but non-zero associations

We should not treat correlation as causality, but this is still enough to justify our current feature strategy: keep all cleaned features in the first version, then test ablations later if needed.

## 8. What these figures change in the modelling pipeline

Each figure informs a specific later step.

- Data quality checks:
  remove impossible records before any training
- Monthly trend:
  use chronological splits and evaluate recent years carefully
- Intraday structure:
  use sequence models that can learn repeated half-hour patterns
- Holiday and weekday effects:
  keep explicit calendar variables in the input set
- Renewable and system features:
  use a multivariate setup rather than a univariate target-only model

This is why our final plan is:

1. target = `tsd`
2. data window = `2019-01-01` to `2024-12-05`
3. dataset = cleaned table
4. features = all available cleaned features
5. main models = `LSTM` and `Transformer`
6. baselines = `CNN` and `ARIMA`

## 9. Variable reference

A full variable table is provided here:

[feature_reference.md](/Users/lihaobo/Desktop/UCL/COMP0197/Group Coursework/EDA%20Report/eda/feature_reference.md)

## Files

- Main script: `EDA Report/eda/generate_eda.py`
- Figures: `EDA Report/eda/figures`
- Summary tables: `EDA Report/eda/outputs`
