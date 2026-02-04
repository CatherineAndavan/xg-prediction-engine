# xG-prediction-engine
An end-to-end xG (Expected Goals) pipeline using XGBoost and FastAPI, reaching 0.7716 AUC on 2022 World Cup data.

# ⚽ Expected Goals (xG) Machine Learning Pipeline

### Performance Highlights
Final Test AUC-ROC: 0.7716

Brier Score: 0.2015

Model: XGBoost (Tuned via RandomizedSearchCV)

# 📊 Exploratory Data Analysis (EDA)
Before modeling, a deep dive into the 2022 World Cup data was conducted (see 01_eda.ipynb).
Based on EDA, Angle and Distance proved to be significant drivers for goals.
<img width="872" height="680" alt="image" src="https://github.com/user-attachments/assets/1d7c442d-03f3-4cc0-98ce-9842ee82f5c4" />

    
Please look into the notebook for other interesting insights!!!

Along with situational features like 'is_header' or 'is_one_on_one', there is improvement in the performance of the model. The final results of the XGBoost model as shown below (taken from train.py), the model validates the findings from the EDA.
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/248f54fd-e37f-45b8-850e-8de0ea9c5d96" />


### How to Run
pip install -r requirements.txt

python src/data_ingestion.py (Ingests 2022 WC Data)

python src/train.py (Trains model and saves to /models)

uvicorn app:app --reload (Starts the Prediction API)
