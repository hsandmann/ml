import matplotlib.pyplot as plt

from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

plt.figure(figsize=(12, 10))

# Carregar o conjunto de dados Iris
iris = load_iris()
x = iris.data
y = iris.target

# Dividir os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de árvore de decisão
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Avaliar o modelo
accuracy = classifier.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
tree.plot_tree(classifier)

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
