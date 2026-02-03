import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.sentiment_model import SentimentAnalyzer
from src.forecasting import SentimentForecaster


st.set_page_config(page_title="Airline Analysis Panel", layout="wide")

@st.cache_resource
def load_resources():
    return SentimentAnalyzer(), SentimentForecaster()

analyzer, forecaster = load_resources()

st.title("📊 Airline Sentiment Trend & Analysis Panel")

# --- SOL PANEL: TEKİL TAHMİN ---
st.sidebar.header("🔍 Quick Prediction")
user_tweet = st.sidebar.text_area("Enter a tweet:", "The flight was okay but delayed.")
if st.sidebar.button("Find Sentiment"):
    res = analyzer.predict(user_tweet)
    st.sidebar.write(f"**Result:** {res['label']} (%{res['confidence']*100:.1f})")

# --- MAIN PANEL: DATA ANALYSIS ---
uploaded_file = st.file_uploader("Upload Tweets.csv file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Historical Negative Trend")
        # Forecasting
        forecast_df = forecaster.forecast(df)
        
        # Plotting
        fig, ax = plt.subplots()
        ax.plot(pd.to_datetime(forecast_df['ds']), forecast_df['yhat'], color='red', label='Forecast')
        ax.fill_between(pd.to_datetime(forecast_df['ds']), forecast_df['yhat_lower'], forecast_df['yhat_upper'], color='pink', alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader("🔮 Future Prediction")
        # Show only the next 7 days
        next_week = forecast_df.tail(7)[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted Complaints'})
        st.dataframe(next_week)