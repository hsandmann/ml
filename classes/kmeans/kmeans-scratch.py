import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.clusters = None

    def fit(self, X):
        if self.random_state:
            np.random.seed(self.random_state)
        
        # Initialize centroids randomly
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            old_centroids = self.centroids.copy()
            self.clusters = self._assign_clusters(X)
            
            # Update centroids
            for i in range(self.k):
                points = X[self.clusters == i]
                if len(points) > 0:
                    self.centroids[i] = np.mean(points, axis=0)
            
            # Check for convergence
            if np.all(old_centroids == self.centroids):
                break
        
        return self
    
    def _assign_clusters(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def predict(self, X):
        return self._assign_clusters(X)

# Example usage

# Generate sample data
np.random.seed(42)
X = np.concatenate([
    np.random.normal(0, 1, (100, 2)),
    np.random.normal(5, 1, (100, 2)),
    np.random.normal(10, 1, (100, 2))
])

# Run K-Means
kmeans = KMeans(k=3, max_iters=100, random_state=42)
kmeans.fit(X)

# Get cluster assignments
labels = kmeans.predict(X)
print("Cluster assignments:", labels)
print("Final centroids:", kmeans.centroids)