**K-Means** Clustering is an unsupervised machine learning algorithm used to partition a dataset into \( K \) distinct, non-overlapping clusters. The algorithm assigns each data point to the cluster with the nearest centroid (mean) based on a distance metric, typically Euclidean distance. It is widely used in data analysis, pattern recognition, and image processing due to its simplicity and efficiency.

## Key Concepts

- **Clusters**: Groups of data points that are similar to each other based on a distance metric.
- **Unsupervised Learning**: The algorithm works without labeled data, identifying patterns based solely on the data's structure.
- **Centroids**: These are the "centers" of the clusters, represented as the mean (average) of all points in a cluster.
- **K**: The number of clusters, which must be specified in advance (e.g., K=3 means dividing data into 3 clusters).
- **Distance Metric**: Typically, Euclidean distance is used to measure how far a data point is from a centroid. The goal is to assign points to the nearest centroid.
- **Objective**: Minimize the within-cluster sum of squares (WCSS), which is the sum of squared distances between each point and its assigned centroid. Mathematically, for a dataset \( X = \{x_1, x_2, \dots, x_n\} \) and centroids \( \mu = \{\mu_1, \mu_2, \dots, \mu_K\} \), the objective is:

    \[
    \arg\min_{\mu} \sum_{i=1}^K \sum_{x \in C_i} \|x - \mu_i\|^2
    \]

    where \( C_i \) is the set of points in cluster \( i \).

K-Means assumes clusters are spherical and equally sized, which may not always hold for real data.


## Algorithm: Step-by-Step
The K-Means algorithm is iterative and consists of the following steps:

1. **Initialization**:

    - Choose the value of K (number of clusters).
    - Randomly select K initial centroids from the dataset. (A common improvement is K-Means++ initialization, which spreads out the initial centroids to avoid poor starting points.)

2. **Assignment Step (Expectation)**:

    - For each data point in the dataset, calculate its distance to all K centroids.
    - Assign the point to the cluster with the closest centroid (using Euclidean distance or another metric).
    - This creates K clusters, where each point belongs to exactly one cluster.

3. **Update Step (Maximization)**:

    - For each cluster, recalculate the centroid as the mean (average) of all points assigned to that cluster.
    - Update the centroids with these new values.

4. **Iteration**:

    - Repeat steps 2 and 3 until one of the stopping criteria is met:
        - Centroids no longer change (or change by less than a small threshold, e.g., 0.001).
        - A maximum number of iterations is reached (to prevent infinite loops).
        - The WCSS decreases minimally between iterations.

5. **Output**:

    - The final centroids and the cluster assignments for each data point.

The algorithm converges because the WCSS is non-increasing with each iteration, but it may converge to a local optimum (not always the global best). Running it multiple times with different initializations helps mitigate this.

## Example 1

Suppose you have a 2D dataset with 5 points: (1,2), (2,1), (5,8), (6,7), (8,6). Let K=2.

- **Initialization**: Randomly pick centroids, say C1=(1,2) and C2=(5,8).
- **Assignment**:
    - (1,2) and (2,1) are closer to C1 → Cluster 1.
    - (5,8), (6,7), (8,6) are closer to C2 → Cluster 2.
- **Update**:
    - New C1 = average of (1,2) and (2,1) = (1.5, 1.5).
    - New C2 = average of (5,8), (6,7), (8,6) = (6.33, 7).
- **Repeat**: Reassign points based on new centroids. This continues until stable.

After convergence, you might end up with two clusters: one around (1.5,1.5) and one around (6.33,7).

## Example 2

=== "Result"

    ```python exec="1" html="1"
    --8<-- "docs/classes/kmeans/kmeans-sklearn.py"
    ```

=== "Code"

    ```python exec="0"
    --8<-- "docs/classes/kmeans/kmeans-sklearn.py"
    ```



## Advantages and Disadvantages
| Aspect          | Advantages                                                                 | Disadvantages                                                              |
|-----------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------|
| **Simplicity** | Easy to understand and implement; computationally efficient (O(n) per iteration). | Sensitive to initial centroid placement; may converge to local optima.     |
| **Scalability**| Works well on large datasets with linear time complexity.                  | Assumes spherical clusters; struggles with non-convex or varying densities. |
| **Output**     | Produces tight, compact clusters; interpretable centroids.                 | Requires predefined K; outliers can skew results.                          |

## Choosing K
- **Elbow Method**: Plot WCSS vs. K and look for the "elbow" where the rate of decrease slows (e.g., K=3 if the curve bends sharply there).
- **Silhouette Score**: Measures how similar points are within their cluster vs. other clusters (higher is better, range -1 to 1).
- Other methods: Gap statistic or domain knowledge.

## Implementation Tip
In Python, you can use scikit-learn's `KMeans` class:
```python
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1,2], [2,1], [5,8], [6,7], [8,6]])  # Your data
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)
print(kmeans.labels_)  # Cluster assignments: e.g., [0, 0, 1, 1, 1]
print(kmeans.cluster_centers_)  # Centroids
```

K-Means is a foundational algorithm, but variants like hierarchical clustering or DBSCAN may be better for certain data types. If you have specific data or code to run, let me know for a demo!

## Additional

### K-Means++ Initialization: Explanation

K-Means++ is an improved initialization method for the K-Means clustering algorithm, designed to address the sensitivity of standard K-Means to the initial placement of centroids. Randomly choosing initial centroids in standard K-Means can lead to poor clustering results or convergence to suboptimal local minima. K-Means++ mitigates this by strategically selecting initial centroids to be spread out across the data, improving both the quality of clusters and convergence speed.

#### Why K-Means++?

In standard K-Means, centroids are often initialized randomly, which can result in:

- **Poor clustering**: Random centroids might be too close to each other, leading to unbalanced or suboptimal clusters.
- **Slow convergence**: Bad initial placements require more iterations to reach a stable solution.
- **Inconsistent results**: Different runs produce varying clusters due to random initialization.

K-Means++ addresses these issues by choosing initial centroids in a way that maximizes their separation, reducing the likelihood of poor starting conditions.

#### K-Means++ Initialization

The K-Means++ algorithm selects the initial K centroids iteratively, using a probabilistic approach that favors points farther from already chosen centroids. Here’s the step-by-step process for a dataset \( X = \{x_1, x_2, \dots, x_n\} \) and \( K \) clusters:

1. **First Centroid**:
    - Randomly select one data point from the dataset as the first centroid \( \mu_1 \). This is typically done uniformly at random to ensure fairness.

2. **Subsequent Centroids**:
    - For each remaining centroid (from 2 to K):
        - Compute the squared Euclidean distance \( D(x) \) from each data point \( x \) to the *nearest* already-selected centroid.
        - Assign a probability to each point \( x \): \( \frac{D(x)^2}{\sum_{x' \in X} D(x')^2} \). Points farther from existing centroids have a higher probability of being chosen.
        - Select the next centroid by sampling a point from the dataset, weighted by these probabilities.
    - This ensures new centroids are likely to be far from existing ones, spreading them across the data.

3. **Repeat**:
    - Continue selecting centroids until all K are chosen.

4. **Proceed to K-Means**:
    - Use these K centroids as the starting point for the standard K-Means algorithm (assign points to nearest centroids, update centroids, iterate until convergence).

#### Mathematical Intuition

The probability function \( \frac{D(x)^2}{\sum D(x')^2} \) uses squared distances to emphasize points that are farther away. This creates a "repulsive" effect, where new centroids are more likely to be placed in regions of the dataset that are not yet covered by existing centroids. The result is a set of initial centroids that are well-distributed, reducing the chance of clustering points into suboptimal groups.

The expected approximation ratio of K-Means++ is \( O(\log K) \)-competitive with the optimal clustering, a significant improvement over random initialization, which has no such guarantee.

#### Example

Suppose you have a dataset with points: (1,1), (2,2), (8,8), (9,9), and you want \( K=2 \):

- **Step 1**: Randomly pick (1,1) as the first centroid.
- **Step 2**: Calculate squared distances to (1,1):

    - (1,1): \( 0^2 = 0 \)
    - (2,2): \( (1^2 + 1^2) = 2 \)
    - (8,8): \( (7^2 + 7^2) = 98 \)
    - (9,9): \( (8^2 + 8^2) = 128 \)
    - Total: \( 0 + 2 + 98 + 128 = 228 \).
    - Probabilities:

        (1,1): \( 0/228 = 0 \),
        
        (2,2): \( 2/228 \approx 0.009 \),
        
        (8,8): \( 98/228 \approx 0.43 \),
        
        (9,9): \( 128/228 \approx 0.56 \).

    - Likely pick (9,9) or (8,8) as the second centroid due to their high probabilities (far from (1,1)).

- **Result**: Centroids like (1,1) and (9,9) are well-spread, leading to better clustering than if (1,1) and (2,2) were chosen.

#### Advantages and Disadvantages

| Aspect          | Advantages                                                                 | Disadvantages                                                              |
|-----------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------|
| **Quality**     | Produces better initial centroids, leading to lower WCSS and better clusters. | Slightly more computationally expensive than random initialization.         |
| **Convergence** | Often converges faster due to better starting points (fewer iterations).    | Still requires predefined K; sensitive to outliers (can skew distances).   |
| **Consistency** | More consistent results across runs compared to random initialization.      | Random first centroid can still introduce some variability.                |

#### Computational Cost

- **Random Initialization**: O(K) for picking K random points.
- **K-Means++**: O(nK) for computing distances to select K centroids, where n is the number of points. This is a small overhead compared to the K-Means iterations (O(nKI), where I is the number of iterations), and the improved clustering quality often outweighs the cost.

#### Implementation
In Python’s scikit-learn, K-Means++ is the default initialization method for the `KMeans` class:
```python
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1,1], [2,2], [8,8], [9,9]])
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42, n_init=10)
kmeans.fit(X)
print(kmeans.labels_)  # Cluster assignments
print(kmeans.cluster_centers_)  # Centroids
```
The `init='k-means++'` parameter explicitly sets K-Means++ initialization (though it’s default in scikit-learn).

#### Practical Notes

- **Choosing K**: K-Means++ still requires you to specify K. Use methods like the elbow method or silhouette score to determine an optimal K.
- **Outliers**: Outliers can disproportionately affect centroid selection due to squared distances. Preprocessing (e.g., removing outliers) can help.
- **Scalability**: For very large datasets, variants like scalable K-Means++ or mini-batch K-Means can be used to reduce computational cost.

K-Means++ is a robust improvement over random initialization, widely used in practice due to its balance of simplicity and effectiveness. If you have a dataset or want a visual demo of K-Means++ vs. random initialization, let me know!

---

## Additional

<iframe width="100%" height="470" src="https://www.youtube.com/embed/njRYKzRKBPY" title="Algoritmo k-means (k-médias)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

## Exercício

!!! success inline end "Entrega"

    :calendar: **21.sep** :clock3: **23:59**

    :material-account: Individual

    :simple-target: Entrega do link via [Canvas](https://canvas.espm.br/){:target="_blank"}.

Dentre os [datasets disponíveis](/ml/classes/concepts/data/main/#datasets){:target="_blank"}, escolha um cujo objetivo seja prever uma variável categórica (classificação). Utilize o algoritmo de K-Means para treinar um modelo e avaliar seu desempenho.

Utilize as bibliotecas `pandas`, `numpy`, `matplotlib` e `scikit-learn` para auxiliar no desenvolvimento do projeto.

A entrega deve ser feita através do [Canvas](https://canvas.espm.br/) - **Exercício K-Means**. Só serão aceitos links para repositórios públicos do GitHub contendo a documentação (relatório) e o código do projeto. Conforme exemplo do [template-projeto-integrador](https://hsandmann.github.io/documentation.template/){:target="_blank"}. ESTE EXERCÍCIO É INDIVIDUAL.

A entrega deve incluir as seguintes etapas:

| Etapa | Critério | Descrição | Pontos |
|:-----:|----------|-----------|:------:|
| 1 | Exploração dos Dados | Análise inicial do conjunto de dados - com explicação sobre a natureza dos dados -, incluindo visualizações e estatísticas descritivas. | 20 |
| 2 | Pré-processamento | Limpeza dos dados, tratamento de valores ausentes e normalização. | 10 |
| 3 | Divisão dos Dados | Separação do conjunto de dados em treino e teste. | 20 |
| 4 | Treinamento do Modelo | Implementação do modelo KNN. | 10 |
| 5 | Avaliação do Modelo | Avaliação do desempenho do modelo utilizando métricas apropriadas. | 20 |
| 6 | Relatório Final | Documentação do processo, resultados obtidos e possíveis melhorias. **Obrigatório:** uso do template-projeto-integrador, individual. | 20 |