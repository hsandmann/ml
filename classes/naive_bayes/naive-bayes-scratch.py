class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_prior = {}
        self.feature_prob = {}
        n_samples, n_features = X.shape
        
        for c in self.classes:
            X_c = X[y == c]
            self.class_prior[c] = len(X_c) / n_samples
            total_count = np.sum(X_c)
            self.feature_prob[c] = (np.sum(X_c, axis=0) + self.alpha) / (total_count + self.alpha * n_features)
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                log_prob = np.log(self.class_prior[c])
                log_prob += np.sum(np.log(self.feature_prob[c]) * x)
                posteriors[c] = log_prob
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

# Example usage with numerical data above
X_train = np.array([[1,2,1], [0,1,2], [3,0,0], [2,1,0]])  # Features: sweet, crunchy, red
y_train = np.array(['Apple', 'Apple', 'Orange', 'Orange'])

nb = MultinomialNB()
nb.fit(X_train, y_train)

X_test = np.array([[2,1,0]])
print(nb.predict(X_test))  # Output: ['Orange']