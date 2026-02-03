import pandas as pd
from prophet import Prophet

class SentimentForecaster:
    def __init__(self):
        
        self.model = Prophet(changepoint_prior_scale=0.05, daily_seasonality=True)

    def prepare_data(self, df):
        """
        Ham veriyi Prophet'in beklediği 'ds' ve 'y' formatına çevirir.
        """
        
        df['tweet_created'] = pd.to_datetime(df['tweet_created']).dt.date
        
        
        daily_neg = df[df['airline_sentiment'] == 'negative'].groupby('tweet_created').size().reset_index()
        
        
        daily_neg.columns = ['ds', 'y']
        return daily_neg

    def forecast(self, df, periods=7):
        """
        Gelecek 'periods' gün kadar tahmin yapar.
        """
        data = self.prepare_data(df)
        
        
        self.model.fit(data)
        
       
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        return forecast
    