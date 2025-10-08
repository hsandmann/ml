Support Vector Machine (SVM) is a supervised machine learning algorithm primarily used for classification tasks, though it can also handle regression. At its core, SVM aims to find the optimal hyperplane that separates data points of different classes with the maximum possible margin. The margin is the distance between the hyperplane and the nearest data points from each class, known as support vectors. By maximizing this margin, SVM promotes better generalization to unseen data.

For linearly separable data, the hyperplane is a straight line (in 2D) or a plane (in higher dimensions). However, real-world data is often not linearly separable. This is where the "kernel trick" comes in—it implicitly maps the data into a higher-dimensional space where it becomes linearly separable, without explicitly computing the transformation. Common kernels include linear, polynomial, radial basis function (RBF), and sigmoid.

SVM can be "hard-margin" (no misclassifications allowed, assuming perfect separability) or "soft-margin" (allows some misclassifications via a regularization parameter C to handle noise or overlaps).

### Algorithm

The SVM algorithm involves solving an optimization problem. In the primal form for linear SVM:

Minimize \( \frac{1}{2} \|w\|^2 + C \sum \xi_i \)

Subject to \( y_i (w \cdot x_i + b) \geq 1 - \xi_i \), \( \xi_i \geq 0 \)

Here, w is the weight vector, ξ_i are slack variables for soft margin, and C is the penalty.

The dual form (used for kernels) is:

Maximize \( \sum \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j) \)

Subject to \( \sum \alpha_i y_i = 0 \), \( 0 \leq \alpha_i \leq C \)

This is solved using quadratic programming techniques like Sequential Minimal Optimization (SMO) or numerical optimizers. Once solved, the decision function is \( f(x) = \sign(\sum \alpha_i y_i K(x_i, x) + b) \).

For training:
1. Compute the kernel matrix K.
2. Solve for α using optimization.
3. Compute b using a support vector.
4. Use the decision function for predictions.

### Vanilla Implementation (Two Classes, Non-Linearly Separated)
Below is a basic from-scratch implementation in Python for a two-class problem with non-linearly separated data (concentric circles). It uses the RBF kernel and solves the dual problem with `scipy.optimize.minimize` (SLSQP method). We generate synthetic data, train the SVM, and plot the decision boundary using `matplotlib.pyplot`.

```python
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Generate concentric circles data (non-linearly separable in 2D)
np.random.seed(0)
num_points = 50
theta = np.linspace(0, 2 * np.pi, num_points)

# Inner circle (class -1)
r_inner = 1
x_inner = r_inner * np.cos(theta) + np.random.normal(0, 0.1, num_points)
y_inner = r_inner * np.sin(theta) + np.random.normal(0, 0.1, num_points)

# Outer circle (class +1)
r_outer = 3
x_outer = r_outer * np.cos(theta) + np.random.normal(0, 0.1, num_points)
y_outer = r_outer * np.sin(theta) + np.random.normal(0, 0.1, num_points)

X = np.vstack((np.column_stack((x_inner, y_inner)), np.column_stack((x_outer, y_outer))))
y = np.hstack((-np.ones(num_points), np.ones(num_points)))

# RBF kernel
def rbf_kernel(x1, x2, sigma=1):
    return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * sigma**2))

# Kernel matrix
def kernel_matrix(X, kernel, sigma):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(X[i], X[j], sigma)
    return K

K = kernel_matrix(X, rbf_kernel, 1)

# Objective function for dual (minimize this for maximization)
P = np.outer(y, y) * K
def objective(alpha):
    return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)

# Constraint: sum alpha_i * y_i = 0
def constraint(alpha):
    return np.dot(alpha, y)

cons = {'type': 'eq', 'fun': constraint}

# Bounds for hard margin (alpha >= 0)
bounds = [(0, None) for _ in range(len(y))]

# Initial guess
alpha0 = np.zeros(len(y))

# Optimize
res = optimize.minimize(objective, alpha0, method='SLSQP', bounds=bounds, constraints=cons)
alpha = res.x

# Find support vectors (alpha > threshold)
sv_threshold = 1e-5
sv_idx = alpha > sv_threshold

# Compute bias b using one support vector
i = np.where(sv_idx)[0][0]
b = y[i] - np.dot(alpha * y, K[i, :])

# Prediction function
def predict(x):
    kx = np.array([rbf_kernel(x, xi, sigma=1) for xi in X])
    return np.dot(alpha * y, kx) + b

# Plot the data and decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = np.array([predict(np.array([r, c])) for r, c in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], colors=['#FFDDDD', '#DDDDFF'], alpha=0.8)
plt.contour(xx, yy, Z, levels=[0], colors='k', linestyles='--')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='Class -1')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Class +1')
plt.scatter(X[sv_idx, 0], X[sv_idx, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')
plt.title('SVM with RBF Kernel on Non-Linear Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```

You can copy and run this code in a Python environment with NumPy, SciPy, and Matplotlib installed. The plot will show red points for the inner circle (class -1), blue for the outer (class +1), a dashed black line as the decision boundary (which forms a closed shape around the inner circle due to the RBF kernel), and circled points as support vectors. The background shading indicates the predicted regions for each class.

For a visual example of what such a plot looks like (using moons data for illustration):



The RBF panel shows a curved boundary separating the intertwined classes.

### Applications

SVM is versatile and used in:

- Image classification (e.g., handwritten digit recognition).
- Text categorization (e.g., spam detection).
- Bioinformatics (e.g., protein classification).
- Finance (e.g., stock trend prediction).
- Face detection in computer vision.

### Pros and Cons

<div class="grid cards" markdown>

-   __Pros__

    ---

    - Effective in high-dimensional spaces.
    - Robust to overfitting, especially with appropriate kernels and C.
    - Memory efficient, as it only relies on support vectors.
    - Versatile with different kernels for non-linear problems.

-   __Cons__

    ---

    - Computationally intensive for large datasets (O(n^2) for kernel matrix).
    - Sensitive to choice of kernel and parameters (requires tuning).
    - Not probabilistic (doesn't output probabilities directly; needs extensions like Platt scaling)[^8].
    - Poor performance on noisy data without proper regularization.

</div>

## Additional

### Video Explanation

<iframe width="100%" height="460" src="https://www.youtube.com/embed/efR1C6CvhmE" title="Support Vector Machines Part 1 (of 3): Main Ideas!!!" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

### Terminology

- **Hyperplane**: The decision boundary that separates classes. In 2D, it's a line; in 3D, a plane.
- **Support Vectors**: The data points closest to the hyperplane that influence its position and orientation. Only these points matter for the model.
- **Margin**: The perpendicular distance from the hyperplane to the support vectors. SVM maximizes this for robustness.
    - **Hard Margin**: No misclassifications allowed; assumes perfect separability.
    - **Soft Margin**: Allows some misclassifications via slack variables (ξ_i) to handle noise/overlap.
- **Kernel Trick**: A method to compute inner products in a high-dimensional space without explicitly mapping data points, enabling non-linear decision boundaries.
- **Linear Kernel**: \( K(x, x') = x \cdot x' \)
- **Polynomial Kernel**: \( K(x, x') = (x \cdot x' + 1)^d \)
- **Radial Basis Function (RBF) Kernel**: \( K(x, x') = \exp(-\gamma \|x - x'\|^2) \
- **Sigmoid Kernel**: \( K(x, x') = \tanh(\kappa x \cdot x' + c) \)
- **Decision Function**: The function used to classify new data points: \( f(x) = \sign(\sum \alpha_i y_i K(x_i, x) + b) \)
- **Kernel Function**: A function that computes the similarity between data points in a transformed feature space (e.g., RBF kernel: \( K(x, x') = \exp(-\gamma \|x - x'\|^2) \)).
- **Lagrange Multipliers (α)**: Variables used in the dual formulation to solve the optimization problem.
- **Regularization Parameter (C)**: Controls the trade-off between maximizing the margin and minimizing classification errors in soft-margin SVM.
- **Bias (b)**: The offset term in the decision function \( f(x) = \sign(w \cdot x + b) \).
- **Dual Problem**: A reformulation of the primal optimization problem, which is easier to solve, especially with kernels.


[^1]: [Support Vector Machines: A Guide for Beginners - QuantStart](https://www.quantstart.com/articles/Support-Vector-Machines-A-Guide-for-Beginners/){:target="_blank"}
[^2]: [An Idiot's Guide to Support Vector Machines (SVMs) - MIT](https://web.mit.edu/6.034/wwwbob/svm.pdf){:target="_blank"}
[^3]: [Understanding Support Vector Machine (SVM) Algorithm from Examples - Medium](https://medium.com/analytics-vidhya/understanding-support-vector-machine-svm-algorithm-from-examples-4b8e2f3f0b1e){:target="_blank"}
[^4]: [Support Vector Machine (SVM) Algorithm - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/){:target="_blank"}
[^5]: [Scikit-learn SVM Tutorial with Python - DataCamp](https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python){:target="_blank"}
[^6]: [Tutorial on Support Vector Machine (SVM)](https://www.tutorialspoint.com/machine_learning/machine_learning_support_vector_machine.htm){:target="_blank"}
[^7]: [Implementing SVM from Scratch in Python - GeeksforGeeks](https://www.geeksforgeeks.org/implementing-svm-from-scratch-in-python/){:target="_blank"}
[^8]: [Platt Scaling](https://en.wikipedia.org/wiki/Platt_scaling){:target="_blank"}

<!-- https://grok.com/c/ffd527f7-0441-41b6-bca3-a11a43dd34b7 -->









---

## Exercício

!!! success inline end "Entrega"

    :calendar: **07.nov** :clock3: **23:59**

    :material-account: Individual

    :simple-target: Entrega do link via [Canvas](https://canvas.espm.br/){:target="_blank"}.

Dentre os [datasets disponíveis](/ml/classes/concepts/data/main/#datasets){:target="_blank"}, escolha um cujo objetivo seja prever uma variável categórica (classificação). Utilize o algoritmo de SVM para treinar um modelo e avaliar seu desempenho.

Utilize as bibliotecas `pandas`, `numpy`, `matplotlib` e `scikit-learn` para auxiliar no desenvolvimento do projeto.

A entrega deve ser feita através do [Canvas](https://canvas.espm.br/) - **Exercício SVM**. Só serão aceitos links para repositórios públicos do GitHub contendo a documentação (relatório) e o código do projeto. Conforme exemplo do [template-projeto-integrador](https://hsandmann.github.io/documentation.template/){:target="_blank"}. ESTE EXERCÍCIO É INDIVIDUAL.

A entrega deve incluir as seguintes etapas:

| Etapa | Critério | Descrição | Pontos |
|:-----:|----------|-----------|:------:|
| 1 | Exploração dos Dados | Análise inicial do conjunto de dados - com explicação sobre a natureza dos dados -, incluindo visualizações e estatísticas descritivas. | 20 |
| 2 | Pré-processamento | Limpeza dos dados, tratamento de valores ausentes e normalização. | 10 |
| 3 | Divisão dos Dados | Separação do conjunto de dados em treino e teste. | 20 |
| 4 | Treinamento do Modelo | Implementação do modelo SVM. | 10 |
| 5 | Avaliação do Modelo | Avaliação do desempenho do modelo utilizando métricas apropriadas. | 20 |
| 6 | Relatório Final | Documentação do processo, resultados obtidos e possíveis melhorias. **Obrigatório:** uso do template-projeto-integrador, individual. | 20 |
