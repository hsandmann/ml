from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups  # Example dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load sample text data
newsgroups = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'soc.religion.christian'])
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
