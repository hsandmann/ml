
**K-Nearest Neighbors (KNN)** is a simple, versatile, and non-parametric machine learning algorithm used for classification and regression tasks. It operates on the principle of similarity, predicting the label or value of a data point based on the majority class or average of its *k* nearest neighbors in the feature space. KNN is intuitive and effective for small datasets or when interpretability is key.

### Key Concepts

- **Instance-Based Learning**: KNN is a lazy learning algorithm, meaning no explicit training phase is required. It stores the entire dataset and performs calculations at prediction time.

- **Distance Metric**: The algorithm measures the distance between data points to identify the nearest neighbors. Common metrics include:

    | Metric | Formula |
    |--------|---------|
    | Euclidean distance | \( \displaystyle \sqrt{\sum_{i=1}^n (x_i - y_i)^2} \) |
    | Manhattan distance | \( \displaystyle \sum_{i=1}^n \|x_i - y_i\| \) |
    | Minkowski distance | \( \displaystyle \left( \sum_{i=1}^n \|x_i - y_i\|^p \right)^{1/p} \) |

- **K Value**: The number of neighbors considered. A small *k* can be sensitive to noise, while a large *k* smooths predictions but may dilute patterns.

- **Decision Rule**:
    - **Classification**: The majority class among the *k* neighbors determines the predicted class.
    - **Regression**: The average (or weighted average) of the *k* neighbors' values is used.

## Mathematical Foundation

KNN relies on distance calculations to find neighbors. For a data point \( x \), the algorithm:
1. Computes the distance to all points in the dataset using a chosen metric (e.g., Euclidean distance).
2. Selects the *k* closest points.
3. For classification, assigns the class with the most votes among the *k* neighbors. For regression, computes the mean of their values.

### Example: Classification

Given a dataset \( D = \{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\} \), where \( x_i \) is a feature vector and \( y_i \) is the class label, predict the class of a new point \( x \):

- Calculate distances \( d(x, x_i) \) for all \( i \).
- Sort distances and select the *k* smallest.
- Count the class labels of these *k* points and assign the majority class to \( x \).

### Weighted KNN
In weighted KNN, neighbors contribute to the prediction based on their distance. Closer neighbors have higher influence, often weighted by the inverse of their distance:

$$ w_i = \frac{1}{d(x, x_i)} $$

For regression, the prediction is:

$$ \hat{y} = \frac{\sum_{i=1}^k w_i y_i}{\sum_{i=1}^k w_i} $$

## Visualizing KNN
To illustrate, consider a 2D dataset with two classes (blue circles and red triangles). For a new point (green star), KNN identifies the *k* nearest points and assigns the majority class.

![KNN Classification Example](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/KnnClassification.svg/1200px-KnnClassification.svg.png){width=70%}
/// caption
Figure: KNN with k=3. The green star is classified based on the majority class (blue circles) among its three nearest neighbors.
/// 

For regression, imagine predicting a continuous value (e.g., house price) based on the average of the *k* nearest houses’ prices.

### Plot: Decision Boundary

KNN’s decision boundary is non-linear and depends on the data distribution. Below is an example of decision boundaries for different *k* values:

![KNN Decision Boundary](https://scikit-learn.org/stable/_images/sphx_glr_plot_classification_001.png)
/// caption
Figure: Decision boundaries for k=1, k=5, and k=15. Smaller k leads to more complex boundaries, while larger k smooths them.
///

## Pros and Cons of KNN
### Pros
- **Simplicity**: Easy to understand and implement.
- **Non-Parametric**: Makes no assumptions about data distribution, suitable for non-linear data.
- **Versatility**: Works for both classification and regression.
- **Adaptability**: Can handle multi-class problems and varying data types with appropriate distance metrics.

### Cons
- **Computational Cost**: Slow for large datasets, as it requires calculating distances for every prediction.
- **Memory Intensive**: Stores the entire dataset, which can be problematic for big data.
- **Sensitive to Noise**: Outliers or irrelevant features can degrade performance.
- **Curse of Dimensionality**: Performance drops in high-dimensional spaces due to sparse data.
- **Choosing K**: Requires careful tuning of *k* and distance metric to balance bias and variance.

## KNN Implementation
Below are two implementations of KNN: one from scratch and one using Python’s scikit-learn library.

### From Scratch
This implementation includes a basic KNN classifier using Euclidean distance.

=== "Result"
    ```python exec="1" html="1"
    --8<-- "docs/classes/knn/knn-scratch.py"
    ```

=== "Code"
    ```python exec="0"
    --8<-- "docs/classes/knn/knn-scratch.py"
    ```

### Using Scikit-Learn


=== "Result"
    ```python exec="1" html="1"
    --8<-- "docs/classes/knn/knn-sklearn.py"
    ```

=== "Code"
    ```python exec="0"
    --8<-- "docs/classes/knn/knn-sklearn.py"
    ```


---

## Exercício

!!! success inline end "Entrega"

    :calendar: **16.sep** :clock3: **23:59**

    :material-account: Individual

    :simple-target: Entrega do link via [Canvas](https://canvas.espm.br/){:target="_blank"}.

Dentre os [datasets disponíveis](/ml/classes/concepts/data/main/#datasets){:target="_blank"}, escolha um cujo objetivo seja prever uma variável categórica (classificação). Utilize o algoritmo de KNN para treinar um modelo e avaliar seu desempenho.

Utilize as bibliotecas `pandas`, `numpy`, `matplotlib` e `scikit-learn` para auxiliar no desenvolvimento do projeto.

A entrega deve ser feita através do [Canvas](https://canvas.espm.br/) - **Exercício KNN**. Só serão aceitos links para repositórios públicos do GitHub contendo a documentação (relatório) e o código do projeto. Conforme exemplo do [template-projeto-integrador](https://hsandmann.github.io/documentation.template/){:target="_blank"}. ESTE EXERCÍCIO É INDIVIDUAL.

A entrega deve incluir as seguintes etapas:

| Etapa | Critério | Descrição | Pontos |
|:-----:|----------|-----------|:------:|
| 1 | Exploração dos Dados | Análise inicial do conjunto de dados - com explicação sobre a natureza dos dados -, incluindo visualizações e estatísticas descritivas. | 20 |
| 2 | Pré-processamento | Limpeza dos dados, tratamento de valores ausentes e normalização. | 10 |
| 3 | Divisão dos Dados | Separação do conjunto de dados em treino e teste. | 20 |
| 4 | Treinamento do Modelo | Implementação do modelo KNN. | 10 |
| 5 | Avaliação do Modelo | Avaliação do desempenho do modelo utilizando métricas apropriadas. | 20 |
| 6 | Relatório Final | Documentação do processo, resultados obtidos e possíveis melhorias. **Obrigatório:** uso do template-projeto-integrador, individual. | 20 |

