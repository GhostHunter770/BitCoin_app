
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
import streamlit as st 
import joblib
#dataframe
# Title
st.title("User Input for Prediction")

st.write("""
### Provide input for the following columns to make a prediction.
""")

# Collect User Inputs
st.sidebar.header("User Inputs")
def getinput():

    # User inputs for numerical columns
    high = st.sidebar.number_input("High", min_value=0.0, step=0.01)
    low = st.sidebar.number_input("Low", min_value=0.0, step=0.01)
    volume = st.sidebar.number_input("Volume", min_value=0.0, step=0.01)
    quote_asset_volume = st.sidebar.number_input("Quote Asset Volume", min_value=0.0, step=0.01)
    number_of_trades = st.sidebar.number_input("Number of Trades", min_value=0, step=1)
    taker_buy_base_asset_volume = st.sidebar.number_input("Taker Buy Base Asset Volume", min_value=0.0, step=0.01)
    taker_buy_quote_asset_volume = st.sidebar.number_input("Taker Buy Quote Asset Volume", min_value=0.0, step=0.01)

    # Categorical inputs
    month = st.sidebar.selectbox("Month", list(range(1, 13)))
    day_name = st.sidebar.selectbox("Day Name", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    hour = st.sidebar.number_input("Hour", min_value=0, max_value=23, step=1)
    quarter = st.sidebar.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"])
    minute = st.sidebar.number_input("Minute", min_value=0, max_value=59, step=1)
    day = st.sidebar.number_input("Day", min_value=1, max_value=31, step=1)
    
    user_data = {
    "High": high,
    "Low": low,
    "Volume": volume,
    "Quote asset volume": quote_asset_volume,
    "Number of trades": number_of_trades,
    "Taker buy base asset volume": taker_buy_base_asset_volume,
    "Taker buy quote asset volume": taker_buy_quote_asset_volume,
    "Month": month,
    "Day_Name": day_name,
    "Hour": hour,
    "Quarter": quarter,
    "Minute": minute,
    "Day": day,
    }
    return pd.DataFrame([user_data])
model = joblib.load('Final_model.h5')
st.write(model.predict(getinput()))
