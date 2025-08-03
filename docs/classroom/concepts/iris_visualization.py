import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from io import StringIO
from sklearn.datasets import load_iris

plt.figure(figsize=(12, 10))

# Carregar o conjunto de dados Iris
iris = load_iris()

# Transforma em DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target_names[iris.target]

# Visualizar o conjunto de dados Iris
sns.pairplot(df, hue='target', height=3)

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())