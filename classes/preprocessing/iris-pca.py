import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Loading Iris dataset
iris = load_iris()

# Transform in dataframe
df = pd.DataFrame(
    data=iris.data,
    columns=['sepal_l', 'sepal_w', 'petal_l', 'petal_w']
)
df['class'] = iris.target_names[iris.target]

X = df.iloc[:,0:4].values
y = df.iloc[:,4].values

# Standardizing
X_std = StandardScaler().fit_transform(X)

# Covariance
cov_mat = np.cov(X_std.T)

# Calculate autovalues and autovectors
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs: print(i[0])

# Sum the cummulative of each eigen value
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

n_eigen = [1, 2, 3, 4]

# Plot the cumulative for each eign value
plt.figure(figsize=(6, 4))
plt.bar(n_eigen, var_exp, alpha=0.5, align='center',
    label='individual explained variance')
plt.step(n_eigen, cum_var_exp, where='mid',
    label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()

# Take the only the two firsts eigen values
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

print('*' * 10)
print('Reduced to 2-D')
print('Matrix W:\n', matrix_w)

# Calculate the new Y for all samples
Y = X_std.dot(matrix_w)

# Plot the data for the 2 firsts principal components
plt.figure(figsize=(6, 4))
for lab, col in zip(('setosa', 'versicolor', 'virginica'), ('blue', 'red', 'green')):
    plt.scatter(Y[y==lab, 0],
                Y[y==lab, 1],
                label=lab,
                c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='lower center')
plt.tight_layout()

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())