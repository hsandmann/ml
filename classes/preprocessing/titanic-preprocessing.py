import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Preprocess the data
def preprocess(df):
    # Fill missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Convert categorical variables
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
    
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    return df[features]

# Load the Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/hsandmann/ml/refs/heads/main/data/kaggle/titanic-dataset.csv')
df = df.sample(n=10)

# Preprocessing
df = preprocess(df)

# Display the first few rows of the dataset
print(df.sample(n=7).to_markdown(index=False))
