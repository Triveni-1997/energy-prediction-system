# Week 4 Findings — Advanced Models

## 1. Models Trained

| # | Model | Method |
|---|-------|--------|
| 1 | XGBoost (baseline) | Default params |
| 2 | LightGBM (baseline) | Default params |
| 3 | LightGBM-Tuned | Optuna (30 trials) |
| 4 | LSTM | 2-layer, 24h sequences |

## 2. Full Model Comparison

| Rank | Model | Test RMSE | Test MAE | Test R² |
|------|-------|-----------|----------|---------|
| 🥇 | **LightGBM-Tuned** | **0.4477** | **0.3159** | **0.6116** |
| 🥈 | LightGBM | 0.4488 | 0.3154 | 0.6097 |
| 🥉 | XGBoost | 0.4515 | 0.3196 | 0.6050 |
| 4 | Gradient Boosting | 0.4525 | 0.3191 | 0.6032 |
| 5 | Random Forest | 0.4561 | 0.3196 | 0.5969 |
| 6 | Linear Regression | 0.4936 | 0.3538 | 0.5278 |
| 7 | LSTM | 0.5513 | 0.4021 | 0.4070 |

## 3. Best Model: LightGBM-Tuned

**Hyperparameters (Optuna best):**
n_estimators: 1000
learning_rate: 0.0102
num_leaves: 69
min_child_samples: 34
subsample: 0.713
colsample_bytree: 0.601
reg_alpha: 1.347
reg_lambda: 0.199

**Why it won:**
- Highest test R², lowest RMSE
- Trains in <2 seconds
- Small train-test gap (0.17)
- Handles missing values natively

## 4. Why LSTM Underperformed

The LSTM achieved only R² = 0.407. Reasons:

1. **Engineered features already encode temporal signal** — lags and rolling means give tree models explicit memory
2. **Dataset too small** (27K samples) for 35K parameters
3. **Noise-dominated** — trees are more robust
4. **Short dependency horizon** — dominant signal is `lag_1h`, which is trivial for trees

This is consistent with academic literature on tabular time series.

## 5. Saved Artifacts for Deployment

| File | Purpose |
|------|---------|
| `models_saved/lightgbm_tuned.pkl` | **Production model** |
| `models_saved/scaler_X.pkl` | Feature scaler |
| `models_saved/scaler_y.pkl` | Target scaler |
| `outputs/model_results.csv` | Comparison table |
| `outputs/plots/07_model_comparison.png` | Visual comparison |

## 6. Key Learnings

1. **Tree models dominate tabular forecasting** (+8% R² vs Linear Regression)
2. **Tuning gives small gains** (LightGBM +0.002)
3. **Deep learning isn't always better** — LSTM lost by 0.20 R²
4. **Feature engineering > model complexity**
5. **Physical leakage is a real risk** — detected and removed `global_intensity`/`global_reactive_power`

## 7. Production Model Selected

**`lightgbm_tuned.pkl`** — best test R², fastest inference, smallest overfit gap.

## 8. Ready for Week 5–6

Next: Flask API + SQLite database + authentication.
