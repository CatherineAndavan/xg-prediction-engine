#!/usr/bin/env python
# coding: utf-8

# In[7]:


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="xG Prediction API")

# Load your 0.7716 AUC model at startup
model = joblib.load('models/xg_model.pkl')

class ShotData(BaseModel):
    dist_center: float
    angle_deg: float
    is_header: int
    is_one_on_one: int

@app.get("/")
def home():
    return {"message": "xG Model API is live. Use /predict to get Expected Goals."}

@app.post("/predict")
def predict_xg(data: ShotData):
    # Converting input to DataFrame for the model
    input_df = pd.DataFrame([data.dict()])

    # Getting the probability for the 'Goal' class (index 1)
    xg_value = model.predict_proba(input_df)[:, 1][0]

    return {
        "expected_goals": round(float(xg_value), 4),
        "tactical_context": "High-threat" if xg_value > 0.3 else "Normal"
    }


# In[ ]:





# In[ ]:




