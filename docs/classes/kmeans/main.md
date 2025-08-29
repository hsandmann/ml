**K-Means** Clustering is an unsupervised machine learning algorithm used to partition a dataset into \( K \) distinct, non-overlapping clusters. The algorithm assigns each data point to the cluster with the nearest centroid (mean) based on a distance metric, typically Euclidean distance. It is widely used in data analysis, pattern recognition, and image processing due to its simplicity and efficiency.

### Key Concepts

- **Clusters**: Groups of data points that are similar to each other based on a distance metric.
- **Centroids**: The mean (or center) of all points in a cluster, used as the representative point for that cluster.
- **Objective**: Minimize the within-cluster sum of squares (WCSS), also known as the inertia, which measures the variance within each cluster.
- **Unsupervised Learning**: The algorithm works without labeled data, identifying patterns based solely on the data's structure.

## Mathematical Foundation

K-Means aims to minimize the following objective function (WCSS):

\[
J = \sum_{i=1}^{n} \sum_{k=1}^{K} w_{ik} \lVert x_i - \mu_k \rVert^2
\]

Where:
- \( n \): Number of data points.
- \( K \): Number of clusters.
- \( x_i \): Data point \( i \).
- \( \mu_k \): Centroid of cluster \( k \).
- \( w_{ik} \): Binary indicator (1 if \( x_i \) belongs to cluster \( k \), 0 otherwise).
- \( \lVert x_i - \mu_k \rVert^2 \): Squared Euclidean distance between point \( x_i \) and centroid \( \mu_k \).

### Algorithm Steps

1. **Initialization**: Randomly select \( K \) initial centroids (often using methods like random selection or K-Means++).
2. **Assignment**: Assign each data point to the nearest centroid based on Euclidean distance.
3. **Update**: Recalculate the centroids as the mean of all points assigned to each cluster.
4. **Iteration**: Repeat steps 2 and 3 until the centroids stabilize (i.e., no significant change) or a maximum number of iterations is reached.
5. **Output**: Return the final clusters and centroids.

## Visualizing K-Means Clustering

### Example Plot: Initial State
Imagine a 2D dataset with points scattered in a plane. Initially, \( K \) centroids are placed randomly.

![Initial Centroids](https://i.imgur.com/example_initial_centroids.png)

*Caption*: Randomly initialized centroids (red stars) among data points.

### Example Plot: After Assignment
Each point is assigned to the nearest centroid, forming initial clusters.

![Cluster Assignment](https://i.imgur.com/example_cluster_assignment.png)

*Caption*: Data points colored by their assigned cluster after the first iteration.

### Example Plot: Final Clusters
After several iterations, the centroids stabilize, and the clusters are well-defined.

![Final Clusters](https://i.imgur.com/example_final_clusters.png)

*Caption*: Final clusters with updated centroids after convergence.

### Elbow Method for Choosing \( K \)

To determine the optimal number of clusters, the Elbow Method plots the WCSS against different values of \( K \). The "elbow" point, where adding more clusters yields diminishing returns, is chosen as the optimal \( K \).

![Elbow Plot](https://i.imgur.com/example_elbow_plot.png)

*Caption*: Elbow plot showing WCSS vs. number of clusters, with the elbow at \( K=3 \).

## Pros and Cons of K-Means Clustering

### Pros
- **Simplicity**: Easy to understand and implement.
- **Efficiency**: Scales well with large datasets, with a time complexity of \( O(n \cdot K \cdot I \cdot d) \), where \( n \) is the number of points, \( K \) is the number of clusters, \( I \) is the number of iterations, and \( d \) is the dimensionality.
- **Versatility**: Works well for spherical or compact clusters.
- **Fast Convergence**: Typically converges quickly, especially with good initialization (e.g., K-Means++).

### Cons
- **Requires Predefined \( K \)**: The number of clusters must be specified beforehand, which may not always be known.
- **Sensitive to Initialization**: Random initialization can lead to suboptimal solutions. K-Means++ mitigates this but doesn't eliminate it.
- **Assumes Spherical Clusters**: Struggles with non-spherical or irregularly shaped clusters.
- **Sensitive to Outliers**: Outliers can skew centroids, affecting cluster quality.
- **Euclidean Distance Limitation**: Relies on Euclidean distance, which may not be suitable for all data types (e.g., categorical data).

## Implementation

Below are two implementations of K-Means clustering: one from scratch and one using a library (scikit-learn).

### From Scratch

This implementation includes the core algorithm with random centroid initialization and Euclidean distance.

=== "Result"
    ```python exec="1" html="1"
    --8<-- "docs/classes/kmeans/kmeans-scratch.py"
    ```

=== "Code"
    ```python exec="0"
    --8<-- "docs/classes/kmeans/kmeans-scratch.py"
    ```

### Using Scikit-Learn


=== "Result"
    ```python exec="1" html="1"
    --8<-- "docs/classes/kmeans/kmeans-sklearn.py"
    ```

=== "Code"
    ```python exec="0"
    --8<-- "docs/classes/kmeans/kmeans-sklearn.py"
    ```

## Conclusion

K-Means Clustering is a powerful yet simple algorithm for partitioning data into meaningful groups. Its mathematical foundation ensures optimization of cluster assignments, but its limitations (e.g., sensitivity to initialization and assumption of spherical clusters) require careful consideration. The provided implementations demonstrate how to apply K-Means both manually and with a robust library like scikit-learn, which includes optimizations like K-Means++. Visualizations and the Elbow Method help in understanding and tuning the algorithm effectively.