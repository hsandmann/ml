from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load example dataset (Iris for classification)
iris = load_iris()
X = iris.data
y = iris.target

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
rf = RandomForestClassifier(n_estimators=100,  # Number of trees
                            max_depth=5,       # Max depth of trees
                            max_features='sqrt',  # Features per split
                            random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
predictions = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")

# Feature importances
print(f"Feature Importances: {rf.feature_importances_}")