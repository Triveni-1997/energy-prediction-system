# Week 3 Findings — Feature Engineering & Baseline Models

## 1. Feature Engineering Summary

| Feature Group | Count | Examples |
|---------------|-------|----------|
| Time | 6 | hour, dayofweek, month, is_weekend |
| Cyclical | 4 | hour_sin/cos, month_sin/cos |
| Lag | 6 | lag_1h, lag_24h, lag_168h |
| Rolling | 4 | rolling_24h_mean/std, rolling_168h_mean/std |
| Weather | 3 | temperature, humidity, wind_speed |
| **Total** | **23** | |

## 2. Data Leakage Fix

**Issue:** Initial model with `global_intensity` + `global_reactive_power` had R²=0.999.

**Cause:** These features are derived from the target at the *same timestamp* (P = V × I).

**Fix:** Removed `global_intensity`, `global_reactive_power`, `voltage`.

**Result:** R² dropped to a realistic ~0.60.

## 3. Model Comparison

| Model | Test RMSE | Test MAE | Test R² | Train-Test Gap |
|-------|-----------|----------|---------|----------------|
| Linear Regression | 0.4936 | 0.3538 | 0.5278 | -0.07 |
| Random Forest | 0.4561 | 0.3196 | 0.5969 | 0.25 (overfit) |
| **Gradient Boosting** | **0.4525** | **0.3191** | **0.6032** | **0.12** ⭐ |

**Best:** Gradient Boosting (highest test R², smallest overfit gap).

## 4. Feature Importance (Random Forest)

Top 5:
1. `lag_1h` (0.64)
2. `lag_168h` (0.04)
3. `hour` (0.03)
4. `lag_24h` (0.03)
5. `hour_sin` (0.03)

**Insight:** Past consumption is the strongest predictor.

## 5. Model Artifacts

- `models_saved/linear_regression.pkl`
- `models_saved/random_forest.pkl`
- `models_saved/gradient_boosting.pkl`
- `outputs/model_results.csv`
- `outputs/plots/07_model_comparison.png`

## 6. Next: Week 4

- Tune Gradient Boosting
- Try XGBoost / LightGBM
- Train LSTM for temporal patterns
- Target R²: 0.70–0.80