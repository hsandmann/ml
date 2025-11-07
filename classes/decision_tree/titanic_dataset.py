import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/hsandmann/ml/refs/heads/main/data/kaggle/titanic-dataset.csv')

print(df.head(5).to_markdown(index=False))