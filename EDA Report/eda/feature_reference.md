# Feature Reference

The table below describes the variables used in the cleaned modelling dataset and the derived calendar features we plan to feed into the models.

| Variable | Type | Role | Description | Why we keep it |
| --- | --- | --- | --- | --- |
| `settlement_date` | timestamp | index / time key | Half-hour timestamp for each observation | Needed for chronological split and feature derivation |
| `settlement_period` | numeric | time feature | Half-hour slot in the day, from 1 to 48 | Captures intraday position directly |
| `period_hour` | numeric / time-like | time feature | Human-readable half-hour time derived from `settlement_period` | Useful for inspection and optional engineered features |
| `tsd` | numeric | target | Transmission System Demand in MW | Main forecasting target |
| `nd` | numeric | input feature | National Demand in MW | Closely related demand signal that may improve multivariate forecasting |
| `england_wales_demand` | numeric | input feature | Electricity demand for England and Wales | Strong demand-side contextual signal |
| `embedded_wind_generation` | numeric | input feature | Embedded wind generation in MW | Helps explain demand variation under changing renewable conditions |
| `embedded_wind_capacity` | numeric | input feature | Installed embedded wind capacity in MW | Captures longer-term system change |
| `embedded_solar_generation` | numeric | input feature | Embedded solar generation in MW | Important for daytime demand behaviour |
| `embedded_solar_capacity` | numeric | input feature | Installed embedded solar capacity in MW | Captures longer-term solar expansion |
| `non_bm_stor` | numeric | input feature | Non-balancing-mechanism storage contribution | Reflects system operation conditions |
| `pump_storage_pumping` | numeric | input feature | Pumped storage demand / pumping activity | Strong system-level explanatory feature |
| `ifa_flow` | numeric | input feature | Flow on the IFA interconnector | Reflects cross-border power exchange |
| `ifa2_flow` | numeric | input feature | Flow on the IFA2 interconnector | Additional interconnector context |
| `britned_flow` | numeric | input feature | Flow on the BritNed interconnector | Additional interconnector context |
| `moyle_flow` | numeric | input feature | Flow on the Moyle interconnector | Additional interconnector context |
| `east_west_flow` | numeric | input feature | Flow on the East-West interconnector | Additional interconnector context |
| `nemo_flow` | numeric | input feature | Flow on the Nemo interconnector | Additional interconnector context |
| `is_holiday` | binary | input feature | Indicator for public holiday | Strong calendar effect on demand level |
| `day_of_week` | derived numeric | input feature | Day of week derived from timestamp | Helps capture weekly seasonality |
| `is_weekend` | derived binary | input feature | Weekend indicator derived from timestamp | Useful because weekday and weekend profiles differ clearly |
| `month` | derived numeric | input feature | Month derived from timestamp | Helps encode annual seasonality |

The raw dataset also contains variables that were removed before modelling because of structural missingness in the earlier years:

| Variable | Status | Reason removed from the cleaned dataset |
| --- | --- | --- |
| `nsl_flow` | removed | Structural missingness in earlier years |
| `eleclink_flow` | removed | Structural missingness in earlier years |
| `scottish_transfer` | removed | Structural missingness in earlier years |
| `viking_flow` | removed | Structural missingness in earlier years |
| `greenlink_flow` | removed | Structural missingness in earlier years |
