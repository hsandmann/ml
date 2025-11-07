import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/hsandmann/ml/refs/heads/main/data/fraude.csv')

print(df.head(20).to_markdown(index=False))