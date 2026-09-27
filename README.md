# Energy Consumption Prediction in Smart Buildings

An ML-based system to forecast hourly energy consumption in smart buildings using historical data, weather, and engineered time-series features.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| ML/DL | scikit-learn, XGBoost, LightGBM, TensorFlow/Keras, Optuna |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Backend | Flask |
| Database | SQLite |
| Frontend | HTML5, CSS3, JavaScript |
| Version Control | Git |

## Project Structure
energy-prediction-system/
├── app/ # Flask app (backend + routes)
├── data/
│ ├── raw/ # Original UCI dataset
│ ├── processed/ # Cleaned + engineered features
│ └── external/ # Weather data
├── models_saved/ # Trained model artifacts
├── notebooks/ # EDA findings (WEEK*_FINDINGS.md)
├── outputs/
│ ├── plots/ # Generated visualizations
│ └── model_results.csv # All model metrics
├── src/
│ ├── data/ # Data loading + weather fetching
│ ├── eda/ # EDA scripts
│ ├── preprocessing/ # Cleaning pipeline
│ ├── features/ # Feature engineering
│ ├── models/ # Training + tuning + prediction
│ └── utils/ # Config, helpers
├── tests/ # Unit tests
├── requirements.txt
└── README.md

## Setup

1. Clone the repo:
git clone <repo-url>
cd energy-prediction-system

2. Create virtual environment:
python -m venv venv
.\venv\Scripts\Activate.ps1 # Windows

3. Install dependencies:
pip install -r requirements.txt

4. Run the pipeline:
python -m src.preprocessing.clean # Clean raw data
python -m src.features.build_features # Engineer features
python -m src.models.split_data # Train/test split
python -m src.models.train_lgbm # Train models
python -m src.models.tune_lgbm # Hyperparameter tuning

5. Run EDA (optional):
python -m src.eda.basic_stats
python -m src.eda.visualize
python -m src.eda.model_comparison

## Dataset

- **Energy:** [UCI Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- 2,075,259 minute-level rows (Dec 2006 – Nov 2010)
- Target: `global_active_power` (kW)
- **Weather:** Meteostat API (Paris, France)

## Model Performance

| Model | Test R² | Test RMSE |
|-------|---------|-----------|
| Linear Regression | 0.528 | 0.494 |
| Random Forest | 0.597 | 0.456 |
| Gradient Boosting | 0.603 | 0.453 |
| XGBoost | 0.605 | 0.451 |
| LightGBM | 0.610 | 0.449 |
| **LightGBM-Tuned (production)** | **0.612** | **0.448** |
| LSTM | 0.407 | 0.551 |

## Progress

- [x] Week 1: Environment setup, project structure, dependencies
- [x] Week 2: Data acquisition, EDA, cleaning pipeline
- [x] Week 3: Feature engineering + baseline models (R²=0.60)
- [x] Week 4: Advanced models (XGBoost, LightGBM, LSTM, Optuna) — R²=0.612
- [ ] Week 5-6: Backend API (Flask + SQLite)
- [ ] Week 7-8: Frontend dashboard
- [ ] Week 9: Integration & testing
- [ ] Week 10: Deployment

## License

Educational project — Acharya Institute of Technology, MCA