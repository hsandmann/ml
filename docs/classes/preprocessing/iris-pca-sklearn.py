import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as pca
from sklearn.preprocessing import StandardScaler

# Loading Iris dataset
iris = load_iris()

# Transform in dataframe
df = pd.DataFrame(
    data=iris.data,
    columns=['sepal_l', 'sepal_w', 'petal_l', 'petal_w']
)
df['class'] = iris.target_names[iris.target]

X = df.iloc[:,0:4].values
y = df.iloc[:,4].values

# Standardizing
X_std = StandardScaler().fit_transform(X)

sklearn_pca = pca(n_components=2)
Y = sklearn_pca.fit_transform(X_std)

# Plot the data for the 2 firsts principal components
plt.figure(figsize=(6, 4))
for lab, col in zip(('setosa', 'versicolor', 'virginica'), ('blue', 'red', 'green')):
    plt.scatter(Y[y==lab, 0],
                Y[y==lab, 1],
                label=lab,
                c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='lower center')
plt.tight_layout()

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()