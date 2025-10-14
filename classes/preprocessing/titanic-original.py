import pandas as pd

# Load the Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/hsandmann/ml/refs/heads/main/data/kaggle/titanic-dataset.csv')
df = df.sample(n=10)

# Display the first few rows of the dataset
print(df.to_markdown(index=False))
