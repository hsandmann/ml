from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from io import StringIO

cancer = load_breast_cancer()
X = cancer.data[:, :2]
y = cancer.target

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))

kernels = {
    'linear': ax1,
    'sigmoid': ax2,
    'poly': ax3,
    'rbf': ax4
}

for k, ax in kernels.items():
    svm = SVC(kernel=k, C=1)
    svm.fit(X, y)

    DecisionBoundaryDisplay.from_estimator(
        svm,
        X,
        response_method="predict",
        alpha=0.8,
        cmap="Pastel1",
        ax=ax
    )
    ax.scatter(
        X[:, 0], X[:, 1], 
        c=y, 
        s=20, edgecolors="k"
    )
    ax.set_title(k)
    ax.set_xticks([]) 
    ax.set_yticks([])

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()