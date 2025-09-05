import pandas as pd
import yfinance as yf

dat = yf.Ticker("AAPL")
df = dat.history(period='1mo')

df['Change'] = df['Close'].pct_change()
df['Z-Volume'] = df['Volume'].apply(lambda x: (x-df['Volume'].mean())/df['Volume'].std())
df['N-Volume'] = df['Volume'].apply(lambda x: (x-df['Volume'].min())/(df['Volume'].max()-df['Volume'].min()))
df['Z-Change'] = df['Change'].apply(lambda x: (x-df['Change'].mean())/df['Change'].std())
df['N-Change'] = df['Change'].apply(lambda x: (x-df['Change'].min())/(df['Change'].max()-df['Change'].min()))
df = df[['Volume', 'N-Volume', 'Z-Volume', 'Change', 'N-Change', 'Z-Change']].dropna()
print(df.head(10).to_markdown())
