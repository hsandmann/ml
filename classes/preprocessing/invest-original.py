import pandas as pd
import yfinance as yf

dat = yf.Ticker("AAPL")
df = dat.history(period='1mo')

df['Change'] = df['Close'].pct_change()
print(df.head(10).to_markdown())
