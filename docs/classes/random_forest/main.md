
Random forests are a popular ensemble learning algorithm in machine learning, primarily used for classification and regression tasks. Introduced by Leo Breiman in 2001, they build upon decision trees by combining multiple trees into a "forest" to improve accuracy, reduce overfitting, and enhance generalization. The key idea is to create diversity among the trees through randomness, which helps in averaging out errors from individual trees.

![](https://media.geeksforgeeks.org/wp-content/uploads/20250627112439534287/Random-forest-algorithm.webp)
/// caption
Random Forest Algorithm. Source: [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/){:target="_blank"}
///

## Key Concepts:

- **Ensemble Method**: Random forests use bagging (bootstrap aggregating), where each decision tree is trained on a random subset of the training data (with replacement). This reduces variance.
- **Random Feature Selection**: At each node split in a tree, only a random subset of features is considered, which decorrelates the trees and further reduces overfitting.
- **Prediction**:
    - For **classification**: The final output is the majority vote (mode) from all trees.
    - For **regression**: The final output is the average (mean) of predictions from all trees.
- **Advantages**: Robust to noise, handles missing values well, provides feature importance scores, and works well with high-dimensional data.
- **Disadvantages**: Can be computationally intensive, less interpretable than single decision trees, and may require tuning hyperparameters like number of trees (`n_estimators`), maximum depth (`max_depth`), and number of features per split (`max_features`).
- **Applications**: Used in finance (credit scoring), healthcare (disease prediction), e-commerce (recommendation systems), and more.

Random forests also offer out-of-bag (OOB) error estimation, where each tree is evaluated on the data not included in its bootstrap sample, providing a built-in cross-validation metric.

## Formulas and Mathematical Foundations

Random forests build on decision trees, where each tree minimizes impurity (e.g., Gini or entropy for classification, MSE for regression) at splits.

1. **Bootstrap Sampling**:

    - Given a dataset \( D \) with \( N \) samples, for each tree \( t = 1 \) to \( T \):
        - Sample \( D_t \) (bootstrap dataset) with replacement from \( D \), typically of size \( N \).

2. **Feature Subset Selection**:

    - At each node, select a random subset of \( m \) features from total \( p \) features (often \( m = \sqrt{p} \) for classification or \( m = p/3 \) for regression).

3. **Tree Construction**:

    - For classification, impurity measures:

        <div class="grid cards" markdown>

        -   __Gini Impurity__

            ---

            $$ G = \sum_{k=1}^K p_k (1 - p_k) $$
            
            where \( p_k \) is the proportion of class \( k \) in the node.

        -   __Entropy__

            ---

            $$ E = -\sum_{k=1}^K p_k \log_2(p_k) $$

            where \( p_k \) is the proportion of class \( k \) in the node.

        </div>

    - Split to minimize weighted impurity:

        $$ \Delta I = I(parent) - \sum_{child} \frac{N_{child}}{N_{parent}} I(child) $$

4. **Ensemble Prediction**:

    - For classification:

        $$ \hat{y} = \arg\max_k \left( \frac{1}{T} \sum_{t=1}^T I(\hat{y}_t = k) \right) $$
        
        where \( I \) is the indicator function, and \( \hat{y}_t \) is the prediction from tree \( t \).

    - For regression:
    
        $$ \hat{y} = \frac{1}{T} \sum_{t=1}^T \hat{y}_t $$

5. **Out-of-Bag (OOB) Error**:

    - For each sample \( i \), predict using only trees where \( i \) was not in the bootstrap sample.
    - OOB error = average error over all samples (e.g., misclassification rate or MSE).

6. **Feature Importance**:

    - Often measured by mean decrease in impurity (MDI): Sum of \( \Delta I \) across all splits using that feature, averaged over trees.
    - Or permutation importance: Decrease in model score when feature values are randomly shuffled.

These formulas ensure the forest's bias-variance tradeoff is optimized, with low bias from deep trees and low variance from averaging.

## From Scratch

Implementing a random forest from scratch requires building a basic decision tree first, then ensembling them. Below is a simplified version for classification using only standard Python (no external libraries like NumPy for arrays—using lists instead). This is for educational purposes; real implementations use optimized libraries.

We'll assume a binary classification problem with features as lists of lists and labels as a list (0 or 1). It uses Gini impurity and random subsets.

```python
--8<-- "docs/classes/random_forest/random-forest-scratch.py"
```

This implementation is basic and not optimized (e.g., no handling for continuous features beyond simple splits, no OOB). For real use, add error handling and optimizations.

## With Library

For practical applications, use scikit-learn's `RandomForestClassifier` or `RandomForestRegressor`. It handles everything efficiently, including parallel tree building.

```python
--8<-- "docs/classes/random_forest/random-forest-sklearn.py"
```

This uses the Iris dataset for demonstration. You can replace it with your data. scikit-learn handles bootstrapping, randomness, and predictions automatically. For regression, swap to `RandomForestRegressor` and use metrics like MSE.

## Additional

<iframe width="100%" height="470" src="https://www.youtube.com/embed/v6VJ2RO66Ag" title="Random Forest Algorithm Clearly Explained!" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


[^1]: [Random Forest Algorithm in Machine Learning](https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/){:target="_blank"}

[^2]: [Random Forest - Simple Explanation](https://williamkoehrsen.medium.com/random-forest-simple-explanation-377895a60d2d){:target="_blank"}