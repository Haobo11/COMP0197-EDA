# EDA Findings for TSD Forecasting (2019-2024)

This note summarises only the findings that are directly relevant to our final setup:

- target variable: `tsd`
- modelling window: `2019-01-01` to `2024-12-05`
- main models: `LSTM` and `Transformer`
- baselines: `CNN` and `ARIMA`
- feature strategy: use all available features in the cleaned dataset

We restrict the dataset to `2019-01-01` onward because this recent window is more consistent operationally, avoids most of the structural missingness problem in earlier years, and is much more practical for CPU-only training.

## Dataset choice

- Frequency: half-hourly
- Target: `tsd`
- Time span used for modelling: `2019-01-01 00:00:00` to `2024-12-05 23:30:00`
- Data size after filtering: `103,956` rows

This window still contains enough data for sequence modelling while keeping the training cost manageable and the feature space more stable.

## 1. Data quality

The raw dataset contains a small number of physically implausible records:

- `32` rows with `settlement_period > 48`
- `479` rows with `tsd = 0`

We do not want the model to learn these artefacts, so the cleaned dataset is the default starting point for all experiments.

The full historical raw dataset also contains structural missingness in late-appearing interconnector features. That is one of the main reasons we do not use pre-2019 data in the core pipeline.

## 2. Trend and regime change

![Monthly trend](figures/figure_02_target_trend.png)

Within the 2019-2024 window, `tsd` still shows a clear downward shift after 2020, on top of strong annual seasonality.

This matters because:

- the data distribution is not fixed
- train/validation/test splits must be chronological
- evaluation should focus on recent holdout periods

For our modelling plan, this supports using stronger sequence models such as LSTM and Transformer rather than relying only on simpler stationary assumptions.

## 3. Intraday and seasonal structure

![Intraday seasonality](figures/figure_03_intraday_seasonality.png)

The series shows a strong daily cycle:

- lowest demand in the early morning
- rapid ramp after around `06:00`
- strongest peak around `17:00-19:00`
- much larger peak magnitude in winter than in summer

This is one of the clearest reasons for using sequence models. LSTM and Transformer should be able to learn this repeated temporal structure better than a simple linear model, while ARIMA provides a useful baseline for comparison on the same signal.

For feature design, this also justifies keeping all available time-related and system-related inputs rather than reducing the feature space too aggressively.

## 4. Calendar effects

![Holiday effect](figures/figure_05_holiday_effect.png)

Holiday demand is consistently lower than non-holiday demand across the whole day. In the current filtered window, the gap around `18:00` is about `4011 MW`.

![Weekday weekend profile](figures/figure_07_weekday_weekend_profile.png)

Weekday and weekend profiles also separate clearly, especially during business and evening hours. The weekday-weekend gap around `18:00` is about `3424 MW`.

This suggests that calendar information should remain in the full feature set. Even if the model is sequence-based, these signals are too systematic to leave implicit.

## 5. Renewable and system features

![Renewables transition](figures/figure_06_renewables_transition.png)

Renewable-related variables and system operation variables remain relevant in the 2019-2024 period. Installed solar capacity continues to grow, wind capacity remains high, and the demand level keeps shifting over time.

This supports our choice to train using all available cleaned features rather than starting from a heavily reduced subset. For LSTM and Transformer, these variables can help explain why similar calendar positions may behave differently across years.

## 6. What this means for modelling

The EDA supports the following final setup:

1. Use `tsd` as the forecasting target.
2. Restrict the main dataset to `2019-01-01` through `2024-12-05`.
3. Train on the cleaned dataset rather than the raw table.
4. Keep all available cleaned features, including calendar and system variables.
5. Use chronological splits only.
6. Treat `LSTM` and `Transformer` as the main models because the series has strong sequential, seasonal, and regime-dependent structure.
7. Use `CNN` and `ARIMA` as baselines:
   - `ARIMA` gives a classical time-series baseline
   - `CNN` gives a lighter deep-learning baseline for local temporal patterns

## Files

- Main script: `EDA Report/eda/generate_eda.py`
- Figures: `EDA Report/eda/figures`
- Summary tables: `EDA Report/eda/outputs`
