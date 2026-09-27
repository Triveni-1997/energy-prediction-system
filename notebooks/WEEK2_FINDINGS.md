# Week 2 Findings — Data Acquisition & EDA

## 1. Dataset Summary

### Energy Dataset (UCI Household Power Consumption)
| Property | Value |
|----------|-------|
| Source | UCI ML Repository |
| Raw rows | 2,075,259 (minute-level) |
| Date range | Dec 16, 2006 – Nov 26, 2010 |
| Cleaned rows | 34,168 (hourly) |
| Target | `global_active_power` (kW) |
| Missing values | 25,979 rows (1.25%) |
| Outliers removed | 14,376 rows (0.7%) |
| Target range (after cleaning) | 0.076 – 5.188 kW |
| Mean / Median | 1.092 / 0.602 kW |

### Weather Dataset (Meteostat)
| Property | Value |
|----------|-------|
| Source | Meteostat API (Paris, FR) |
| Records | 34,585 hourly |
| Columns | temperature, humidity, precipitation, wind_speed |
| Missing | 60 rows (temp/humidity/wind), 11,991 (precipitation — no rain) |

## 2. Patterns Identified

### Hourly Pattern
- Peak consumption: 7–9 AM and 6–9 PM (morning + evening peaks)
- Lowest consumption: 2–5 AM (sleep hours)
- Clear bimodal daily shape

### Weekly Pattern
- Weekend consumption differs from weekdays
- (Fill in from plot `03_weekly_pattern.png`)

### Seasonal Pattern
- Higher consumption in winter months (Dec–Feb)
- Lower consumption in summer months (Jun–Aug)
- Strong 12-month cycle visible in `01_daily_consumption.png`

### Distribution
- Right-skewed: mean (1.09) > median (0.60)
- Most hours are low-power; occasional high-consumption hours
- Max capped at 5.19 kW after IQR-based outlier removal

## 3. Data Quality Issues

| Issue | Action Taken |
|-------|--------------|
| Missing target values (1.25%) | Dropped rows |
| Duplicate timestamps | None found |
| Extreme outliers (>5 kW) | Capped via IQR × 3 |
| Precipitation missing (34%) | Kept as NaN (means "no rain") |

## 4. Modeling Implications

- **Resample to hourly**: reduces noise, speeds up training (2M → 34K rows)
- **Feature candidates**:
  - Time: hour, dayofweek, month, is_weekend
  - Lag: t-1h, t-24h, t-168h (1 week ago)
  - Rolling: 24h mean, 7d mean
  - Weather: temperature, humidity
- **Target transform**: consider log1p due to right skew
- **Train/test split**: time-based (first 80% train, last 20% test)

## 5. Ready for Week 3

- ✅ Cleaned dataset saved at `data/processed/energy_clean.csv`
- ✅ Weather data at `data/external/weather.csv`
- ✅ Loader functions in `src/data/load_data.py`
- ✅ Cleaning pipeline in `src/preprocessing/clean.py`

**Next:** Feature engineering (`src/features/build_features.py`) and baseline models.