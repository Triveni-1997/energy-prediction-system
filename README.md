# Energy Consumption Prediction in Smart Buildings

An ML-based system to forecast energy consumption in smart buildings using historical data, weather, and occupancy patterns.

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| ML/DL | scikit-learn, TensorFlow/Keras, XGBoost |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Backend | Flask |
| Database | SQLite |
| Frontend | HTML5, CSS3, JavaScript |
| Version Control | Git |

## 📁 Project Structure
energy-prediction-system/
├── app/ # Flask app (backend + routes)
├── data/
│ ├── raw/ # Original UCI dataset
│ ├── processed/ # Cleaned hourly data
│ └── external/ # Weather data
├── notebooks/ # EDA notebooks + findings
├── outputs/plots/ # Generated PNG plots
├── src/
│ ├── data/ # Data loading + weather fetching
│ ├── eda/ # EDA scripts
│ ├── preprocessing/# Cleaning pipeline
│ ├── features/ # Feature engineering
│ ├── models/ # Training, prediction, evaluation
│ └── utils/ # Config, helpers
├── tests/ # Unit tests
├── models_saved/ # Trained model artifacts
├── requirements.txt
└── README.md

## 🚀 Setup

1. Clone the repo
2. Create virtual environment:
python -m venv venv
.\venv\Scripts\Activate.ps1 # Windows
3. Install dependencies:
pip install -r requirements.txt
4. Run EDA:
python -m src.eda.basic_stats
python -m src.eda.visualize
5. Run cleaning pipeline:
python -m src.preprocessing.clean

## 📊 Dataset

- **Energy:** [UCI Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- 2,075,259 minute-level rows (Dec 2006 – Nov 2010)
- Target: `global_active_power` (kW)
- **Weather:** Meteostat API (Paris, France)

## 📈 Progress

- [x] **Week 1**: Environment setup, project structure, dependencies
- [x] **Week 2**: Data acquisition, EDA, cleaning pipeline
- [ ] Week 3: Feature engineering & baseline models
- [x] Week 3: Feature engineering + baseline models (R²=0.60)
- [ ] Week 4: Advanced models (Random Forest, LSTM)
- [ ] Week 5-6: Backend API (Flask)
- [ ] Week 7-8: Frontend dashboard
- [ ] Week 9: Integration & testing
- [ ] Week 10: Deployment

## 📄 License

Educational project — Acharya Institute of Technology, MCA