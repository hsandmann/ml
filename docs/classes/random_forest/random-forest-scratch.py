import random
from collections import Counter

# Helper: Calculate Gini impurity
def gini_impurity(y):
    if not y:
        return 0
    counts = Counter(y)
    impurity = 1
    for count in counts.values():
        prob = count / len(y)
        impurity -= prob ** 2
    return impurity

# Helper: Split dataset based on feature index and value
def split_dataset(X, y, feature_idx, value):
    left_X, left_y, right_X, right_y = [], [], [], []
    for i in range(len(X)):
        if X[i][feature_idx] <= value:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])
    return left_X, left_y, right_X, right_y

# Decision Tree Node
class Node:
    def __init__(self, feature_idx=None, value=None, left=None, right=None, label=None):
        self.feature_idx = feature_idx
        self.value = value
        self.left = left
        self.right = right
        self.label = label  # Leaf node label

# Build a single decision tree (recursive)
def build_tree(X, y, max_depth, min_samples_split, max_features):
    if len(y) < min_samples_split or max_depth == 0:
        return Node(label=Counter(y).most_common(1)[0][0])
    
    n_features = len(X[0])
    features = random.sample(range(n_features), max_features)  # Random subset
    
    best_gini = float('inf')
    best_feature_idx, best_value = None, None
    best_left_X, best_left_y, best_right_X, best_right_y = None, None, None, None
    
    for feature_idx in features:
        values = sorted(set(row[feature_idx] for row in X))
        for value in values:
            left_X, left_y, right_X, right_y = split_dataset(X, y, feature_idx, value)
            if not left_y or not right_y:
                continue
            p_left = len(left_y) / len(y)
            gini = p_left * gini_impurity(left_y) + (1 - p_left) * gini_impurity(right_y)
            if gini < best_gini:
                best_gini = gini
                best_feature_idx = feature_idx
                best_value = value
                best_left_X, best_left_y = left_X, left_y
                best_right_X, best_right_y = right_X, right_y
    
    if best_gini == float('inf'):
        return Node(label=Counter(y).most_common(1)[0][0])
    
    left = build_tree(best_left_X, best_left_y, max_depth - 1, min_samples_split, max_features)
    right = build_tree(best_right_X, best_right_y, max_depth - 1, min_samples_split, max_features)
    return Node(best_feature_idx, best_value, left, right)

# Predict with a single tree
def predict_tree(node, x):
    if node.label is not None:
        return node.label
    if x[node.feature_idx] <= node.value:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)

# Random Forest class
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
    
    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        max_features = int(n_features ** 0.5) if self.max_features == 'sqrt' else self.max_features
        
        for _ in range(self.n_estimators):
            # Bootstrap sample
            bootstrap_idx = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
            X_boot = [X[i] for i in bootstrap_idx]
            y_boot = [y[i] for i in bootstrap_idx]
            tree = build_tree(X_boot, y_boot, self.max_depth, self.min_samples_split, max_features)
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = []
        for x in X:
            tree_preds = [predict_tree(tree, x) for tree in self.trees]
            predictions.append(Counter(tree_preds).most_common(1)[0][0])
        return predictions

# Example usage
# X = [[0, 0], [1, 1], [1, 0], [0, 1]]  # Features
# y = [0, 1, 0, 1]  # Labels
# rf = RandomForest(n_estimators=3, max_depth=2)
# rf.fit(X, y)
# print(rf.predict([[0.5, 0.5]]))  # Output: [1] or similar