#!/usr/bin/env python
# coding: utf-8

# In[2]:


from statsbombpy import sb
import joblib
import os

def ingest():
    # Competition 43 = World Cup, Season 106 = 2022
    print("Streaming 2022 World Cup data...")
    events = sb.competition_events(
        country="International", 
        division="FIFA World Cup", 
        season="2022"
    )

    os.makedirs('data', exist_ok=True)
    # Save as a pickle file to preserve complex columns
    joblib.dump(events, 'data/wc_2022_raw.pkl')
    print("Success: Data frozen in data/wc_2022_raw.pkl")

if __name__ == "__main__":
    ingest()


# In[ ]:




