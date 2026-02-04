# xG-prediction-engine
An end-to-end xG (Expected Goals) pipeline using XGBoost and FastAPI, reaching 0.7716 AUC on 2022 World Cup data.

# ⚽ Expected Goals (xG) Machine Learning Pipeline

### Performance Highlights
Final Test AUC-ROC: 0.7716

Brier Score: 0.2015

Model: XGBoost (Tuned via RandomizedSearchCV)

### How to Run
pip install -r requirements.txt

python src/data_ingestion.py (Ingests 2022 WC Data)

python src/train.py (Trains model and saves to /models)

uvicorn app:app --reload (Starts the Prediction API)
