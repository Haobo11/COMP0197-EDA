# EDA for Assessed Component 2

This package is built for the `Assessed Component 2` progress presentation. The first page of the coursework brief says the presentation should justify the design rationale, show the current working pipeline, and explain technical challenges. The EDA below is structured to support those three points directly.

## Suggested slide flow

### Slide 1: Dataset overview and modelling target
- Dataset: half-hourly UK electricity demand records from 2009-01-01 to 2024-12-05.
- Core targets already visible in the data: `nd` and `tsd`.
- Supporting covariates: embedded wind / solar generation and capacity, interconnector flows, storage behaviour, and holiday flags.
- Presentation line: "Our forecasting problem is strongly temporal and multivariate, so EDA is mainly used to decide what time features, exogenous variables, and preprocessing strategy the PyTorch model will need."

### Slide 2: Raw data quality and preprocessing rationale
- Use [figure_01_missingness_full.png](/Users/lihaobo/Desktop/UCL/COMP0197/Group Coursework/eda/figures/figure_01_missingness_full.png).
- Main point: missingness in the raw dataset is not random. It is concentrated in interconnector variables such as `greenlink_flow`, `viking_flow`, `scottish_transfer`, `nsl_flow`, and `eleclink_flow`.
- Interpretation: these are structural gaps caused by assets appearing later in history, not sensor noise.
- Presentation line: "This tells us we cannot treat all NaNs the same way. For early years, some features simply did not exist, so our preprocessing needs either a clean reduced feature set or a feature availability strategy."

### Slide 3: Long-term trend and non-stationarity
- Use [figure_02_target_trend.png](/Users/lihaobo/Desktop/UCL/COMP0197/Group Coursework/eda/figures/figure_02_target_trend.png) and [figure_04_yearly_distribution.png](/Users/lihaobo/Desktop/UCL/COMP0197/Group Coursework/eda/figures/figure_04_yearly_distribution.png).
- Main point: both `nd` and `tsd` show strong seasonality and a clear regime shift after 2020.
- Interpretation: this is a non-stationary forecasting problem, so we should use chronological train / validation / test splits rather than random splits.
- Presentation line: "The distribution moves over time, so a model that only memorises one operating regime will generalise poorly. That is why our evaluation will be strictly time-aware."

### Slide 4: Daily and seasonal structure
- Use [figure_03_intraday_seasonality.png](/Users/lihaobo/Desktop/UCL/COMP0197/Group Coursework/eda/figures/figure_03_intraday_seasonality.png).
- Main point: demand is lowest overnight, rises through the morning, and peaks around 17:00-19:00, with much larger peaks in winter.
- Interpretation: the model will need explicit calendar / clock features such as hour-of-day, day-of-week, month, and possibly cyclical encodings.
- Presentation line: "This plot is the clearest evidence that the task is not just generic regression. Time position inside the day and season is one of the strongest signals."

### Slide 5: Calendar effects
- Use [figure_05_holiday_effect.png](/Users/lihaobo/Desktop/UCL/COMP0197/Group Coursework/eda/figures/figure_05_holiday_effect.png).
- Main point: holidays shift the full demand profile downward, including evening peak periods.
- Interpretation: holiday indicators are useful exogenous inputs and should not be dropped as a minor feature.
- Presentation line: "Holiday behaviour changes the baseline level of demand, so the network should receive holiday information explicitly rather than trying to infer it from recent lags alone."

### Slide 6: Renewable transition and modelling implications
- Use [figure_06_renewables_transition.png](/Users/lihaobo/Desktop/UCL/COMP0197/Group Coursework/eda/figures/figure_06_renewables_transition.png).
- Main point: embedded wind and solar capacity increase substantially over the historical window while average demand trends downward.
- Interpretation: the data-generating process changes over time; exogenous renewable features matter, and normalisation should be fit only on the training window.
- Presentation line: "This makes the forecasting problem harder but also motivates a multivariate PyTorch model: renewable generation and installed capacity help explain why later years behave differently from earlier years."

## Short speaking script

Our EDA focuses on three questions that matter for the model design. First, what data quality issues are real preprocessing problems and what missing values are structural. Second, which temporal patterns are dominant enough that the model must represent them explicitly. Third, whether the demand process is stable over time or drifting.

The raw-data missingness chart shows that most missing values come from interconnector features that were introduced later, so we treat this as feature availability over time rather than random corruption. After moving to the clean dataset, the monthly trend and yearly distribution plots show clear seasonality and distribution shift, especially after 2020, which justifies a chronological split. The seasonal daily-profile plot shows a strong evening peak and stronger winter demand, so our PyTorch pipeline should include lag features plus calendar features. The holiday plot shows that demand drops on holidays across the whole day, so holiday flags should remain in the model. Finally, renewable capacity growth indicates that the system itself changes over time, which supports using a multivariate forecasting setup rather than a purely univariate baseline.

## Running the code

Use a Python environment with `pandas`, `numpy`, and `matplotlib` installed.

```bash
MPLCONFIGDIR=$PWD/.mplconfig /Applications/miniconda3/bin/python eda/generate_eda.py
```

If your group uses a micromamba environment, the equivalent command is:

```bash
MPLCONFIGDIR=$PWD/.mplconfig micromamba run -n <your-env-name> python eda/generate_eda.py
```

## Files produced

- Figures: `/Users/lihaobo/Desktop/UCL/COMP0197/Group Coursework/eda/figures`
- Tables and summary JSON: `/Users/lihaobo/Desktop/UCL/COMP0197/Group Coursework/eda/outputs`
