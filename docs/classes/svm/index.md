Support Vector Machine (SVM) is a supervised machine learning algorithm primarily used for classification tasks, though it can also handle regression. At its core, SVM aims to find the optimal hyperplane that separates data points of different classes with the maximum possible margin. The margin is the distance between the hyperplane and the nearest data points from each class, known as support vectors. By maximizing this margin, SVM promotes better generalization to unseen data.



For linearly separable data, the hyperplane is a straight line (in 2D) or a plane (in higher dimensions). However, real-world data is often not linearly separable. This is where the "kernel trick" comes in—it implicitly maps the data into a higher-dimensional space where it becomes linearly separable, without explicitly computing the transformation. Common kernels include linear, polynomial, radial basis function (RBF), and sigmoid.

![](./svm_details.png){width=100%}

SVM can be "hard-margin" (no misclassifications allowed, assuming perfect separability) or "soft-margin" (allows some misclassifications via a regularization parameter C to handle noise or overlaps).

## Kernel Trick

The kernel trick allows SVM to operate in a high-dimensional space without explicitly computing the coordinates of the data in that space. Instead, it computes the inner products between all pairs of data points in the original space using a kernel function. This enables SVM to find non-linear decision boundaries efficiently.

=== "1D to 2D"

    ```python exec="1" html="1"
    --8<-- "docs/classes/svm/svm_kernel_transformation.py"
    ```

=== "2D to 3D"

    ![](./svm_kernel_2d_to_3d_example.png)

Common kernel functions include:

| Kernel | Equation |
|---|---|
| Linear | \( K(x,y) = x \cdot y \) |
| Sigmoid | \( K(x,y) = \tanh(ax \cdot y + b) \) |
| Polynomial | \( K(x,y) = (1 + x \cdot y)^d \) |
| Radial Basis Function (RBF) | \( K(x,y) = e^{(-\gamma \|x-y\|^2)} \) |

=== "Result"
    ```python exec="1" html="1"
    --8<-- "docs/classes/svm/svm_breast_cancer.py"
    ```

=== "Code"
    ```python exec="0"
    --8<-- "docs/classes/svm/svm_breast_cancer.py"
    ```

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

---

### Terminology

- **Hyperplane**: The decision boundary that separates classes. In 2D, it's a line; in 3D, a plane.
- **Support Vectors**: The data points closest to the hyperplane that influence its position and orientation. Only these points matter for the model.
- **Margin**: The perpendicular distance from the hyperplane to the support vectors. SVM maximizes this for robustness.
    - **Hard Margin**: No misclassifications allowed; assumes perfect separability.
    - **Soft Margin**: Allows some misclassifications via slack variables (ξ_i) to handle noise/overlap.
- **Kernel Trick**: A method to compute inner products in a high-dimensional space without explicitly mapping data points, enabling non-linear decision boundaries.
- **Decision Function**: The function used to classify new data points:

    \[ f(x) = \sum \alpha_i y_i K(x_i, x) + b) > 0 \]

- **Kernel Function**: A function that computes the similarity between data points in a transformed feature space.
- **Lagrange Multipliers (α)**: Variables used in the dual formulation to solve the optimization problem.
- **Regularization Parameter (C)**: Controls the trade-off between maximizing the margin and minimizing classification errors in soft-margin SVM.
- **Bias (b)**: The offset term in the decision function \( f(x) = w \cdot x + b > 0 \).
- **Dual Problem**: A reformulation of the primal optimization problem, which is easier to solve, especially with kernels.



## Additional


=== "Support Vector Machines: All you need to know!"

    <iframe width="100%" height="470" src="https://www.youtube.com/embed/ny1iZ5A8ilA" title="Support Vector Machines: All you need to know!" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

=== "Support Vector Machines | ML-005 Lecture 12 | Stanford University | Andrew Ng"

    <iframe width="100%" height="470" src="https://www.youtube.com/embed/uV5TnFc7eaE" title="Support Vector Machines | ML-005 Lecture 12 | Stanford University | Andrew Ng" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

=== "Support Vector Machines Part 1 (of 3): Main Ideas!!!"

    <iframe width="100%" height="460" src="https://www.youtube.com/embed/efR1C6CvhmE" title="Support Vector Machines Part 1 (of 3): Main Ideas!!!" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


[^1]: [The Nature of Statistical Learning Theory, Vapnik, 1999](1999 - The Nature of Statistical Learning Theory - Vapnik.pdf){:target="_blank"}.
[^2]: [Support Vector Machines: A Simple Explanation](https://www.kdnuggets.com/2016/07/support-vector-machines-simple-explanation.html){:target="_blank"}
[^3]: [Support Vector Machines: A Guide for Beginners - QuantStart](https://www.quantstart.com/articles/Support-Vector-Machines-A-Guide-for-Beginners/){:target="_blank"}
[^4]: [Support Vector Machine (SVM) Algorithm - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/){:target="_blank"}
[^5]: [Scikit-learn SVM Tutorial with Python - DataCamp](https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python){:target="_blank"}
[^6]: [Tutorial on Support Vector Machine (SVM)](https://www.tutorialspoint.com/machine_learning/machine_learning_support_vector_machine.htm){:target="_blank"}
[^7]: [Implementing SVM from Scratch in Python - GeeksforGeeks](https://www.geeksforgeeks.org/implementing-svm-from-scratch-in-python/){:target="_blank"}
[^8]: [Multi-class classification using Support Vector Machines (SVM)](https://www.geeksforgeeks.org/machine-learning/multi-class-classification-using-support-vector-machines-svm/){:target="_blank"}

---

## Exercício

!!! success inline end "Entrega"

    :calendar: **04.dez** :clock3: **23:59**

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
