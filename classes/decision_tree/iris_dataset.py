import pandas as pd
from sklearn.datasets import load_iris

# Carregar o conjunto de dados Iris
iris = load_iris()

# Transforma em DataFrame
df = pd.DataFrame(
    data=iris.data,
    columns=['sepal_l', 'sepal_w', 'petal_l', 'petal_w']
)
df['class'] = iris.target_names[iris.target]

# Imprime os dados
print(df.sample(frac=.1).to_markdown(index=False))