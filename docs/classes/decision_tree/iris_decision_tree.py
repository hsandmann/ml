import matplotlib.pyplot as plt
import pandas as pd

from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

plt.figure(figsize=(12, 10))

# Carregar o conjunto de dados Iris
iris = load_iris()
x = iris.data
y = iris.target

# Dividir os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo de árvore de decisão
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Evaluate the model
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Optional: Print feature importances
feature_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': classifier.feature_importances_
})
print("<br>Feature Importances:")
print(feature_importance.sort_values(by='Importance', ascending=False).to_html())

tree.plot_tree(classifier)

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())