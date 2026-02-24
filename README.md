# Cricket Match Prediction System

**Predicting IPL match outcomes using ensemble machine learning with real-time data integration**

**Accuracy: >90% on held-out test data**

---

## Project Overview

This project builds an end-to-end machine learning pipeline that predicts IPL cricket match outcomes in real time. It combines ensemble modelling, feature engineering on historical match data, and a live Flask web application — covering the full data science lifecycle from raw data to a deployed prediction API.

---

## Problem Statement

Sports outcome prediction is a high-noise, high-variance problem. Standard single models fail to capture the complex interplay between venue conditions, team form, and player performance. This project solves that by:

- Engineering cricket-specific features from raw match data
- Stacking multiple gradient-boosting models to reduce variance
- Serving real-time predictions via a web interface backed by a live cricket data API

---

## Features Engineered

| Feature | Description |
|---|---|
| `venue_win_ratio` | Historical win % for each team at a given venue |
| `toss_win_ratio` | Impact of toss result on match outcome |
| `player_impact_score` | Composite batting + bowling index per player |
| `head_to_head` | Historical win/loss ratio between the two teams |
| `recent_form` | Rolling average of last 5 match results |

---

## Model Performance

| Model | Accuracy |
|---|---|
| Random Forest | 86% |
| XGBoost | 88% |
| LightGBM | 89% |
| CatBoost | 88% |
| **Ensemble (Voting Classifier)** | **>90%** |

---

## Tech Stack

- **Language:** Python
- **ML Libraries:** Scikit-learn, XGBoost, LightGBM, CatBoost
- **Web Framework:** Flask
- **Data Source:** CricketData API (live ingestion)
- **CI/CD:** GitHub Actions (auto-runs prediction tests on every push)
- **Visualisation:** Power BI, Matplotlib

---

## Project Structure

```
CricketMatchPrediction/
├── .github/workflows/     # CI pipeline
├── Dataset/               # Historical IPL match data
├── notebook/              # EDA and model training notebooks
├── src/                   # Core source code
├── app.py                 # Flask application
├── frontend.html          # Web UI
└── requirements.txt       # Dependencies
```

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/zalakkkkk05/CricketMatchPrediction.git
cd CricketMatchPrediction

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Add your CricketData API key to .env

# Run the app
python app.py
```

Visit `http://localhost:5000`

---

## Key Learnings

- Ensemble methods significantly outperform single models on noisy sports data
- Domain-specific feature engineering contributed more to accuracy than model tuning alone
- CI/CD automation ensures prediction integrity is tested on every code change

---

## Author

**Zalak Patel** — Data Analyst & ML Engineer

[LinkedIn](https://linkedin.com/in/zalak-patel-2989621a1) | [GitHub](https://github.com/zalakkkkk05) | pzalak1234@gmail.com
