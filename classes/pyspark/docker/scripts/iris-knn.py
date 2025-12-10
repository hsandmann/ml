from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
import requests
import pandas as pd
from io import StringIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize Spark session
spark = SparkSession.builder.appName("IrisKNN").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")  # Suppress INFO logs

print("=== Loading Iris Dataset ===")

# Download Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
response = requests.get(url)
data = pd.read_csv(StringIO(response.text), 
                   names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# Convert to Spark DataFrame
df = spark.createDataFrame(data)

print(f"Dataset loaded: {df.count()} rows")
df.show(5)

# Prepare features and label
print("\n=== Preparing Features ===")
assembler = VectorAssembler(
    inputCols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    outputCol='features'
)
indexer = StringIndexer(inputCol='class', outputCol='label')

# Transform data
df_assembled = assembler.transform(df)
df_final = indexer.fit(df_assembled).transform(df_assembled)

print("Feature preparation complete")

# Split data
train, test = df_final.randomSplit([0.8, 0.2], seed=42)

print(f"Training set: {train.count()} rows")
print(f"Test set: {test.count()} rows")

# Convert Spark DataFrames to Pandas for scikit-learn, including original columns
train_pd = train.select('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class', 'features', 'label').toPandas()
test_pd = test.select('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class', 'features', 'label').toPandas()

# Extract features and labels from Pandas DataFrames
X_train = train_pd['features'].tolist()  # Convert features column to list
y_train = train_pd['label']
X_test = test_pd['features'].tolist()
y_test = test_pd['label']

# Train k-NN model
print("\n=== Training k-NN (k=3) ===")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

print("Model trained successfully!")

# Make predictions
print("=== Predictions on Test Set ===")
y_pred = knn.predict(X_test)

# Add predictions to test DataFrame
test_pd['prediction'] = y_pred
test_spark = spark.createDataFrame(test_pd[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class', 'label', 'prediction']])
test_spark.show(10, truncate=False)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("\n=== Model Performance ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Class names mapping
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
print(f"\nClass mapping: {dict(zip(range(3), class_names))}")

spark.stop()
print("\n=== Analysis Complete ===")