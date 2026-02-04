{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "44d896db",
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastapi import FastAPI\n",
    "from pydantic import BaseModel\n",
    "import joblib\n",
    "import pandas as pd\n",
    "\n",
    "app = FastAPI(title=\"xG Prediction API\")\n",
    "\n",
    "# Load your 0.7716 AUC model at startup\n",
    "model = joblib.load('models/xg_model.pkl')\n",
    "\n",
    "class ShotData(BaseModel):\n",
    "    dist_center: float\n",
    "    angle_deg: float\n",
    "    is_header: int\n",
    "    is_one_on_one: int\n",
    "\n",
    "@app.get(\"/\")\n",
    "def home():\n",
    "    return {\"message\": \"xG Model API is live. Use /predict to get Expected Goals.\"}\n",
    "\n",
    "@app.post(\"/predict\")\n",
    "def predict_xg(data: ShotData):\n",
    "    # Converting input to DataFrame for the model\n",
    "    input_df = pd.DataFrame([data.dict()])\n",
    "    \n",
    "    # Getting the probability for the 'Goal' class (index 1)\n",
    "    xg_value = model.predict_proba(input_df)[:, 1][0]\n",
    "    \n",
    "    return {\n",
    "        \"expected_goals\": round(float(xg_value), 4),\n",
    "        \"tactical_context\": \"High-threat\" if xg_value > 0.3 else \"Normal\"\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "acfb9576",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "11d4812b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
