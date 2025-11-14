from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
import numpy as np
from io import StringIO

cancer = load_breast_cancer()
X = cancer.data[:, :2]
y = cancer.target

fig, ax = plt.subplots(2, 1, figsize=(6, 6))

ini = 10
m1 = 20
m2 = 45
end = 75
p = 2

x = np.concatenate((
    np.random.uniform(ini, m1, size=10),
    np.random.uniform(m1, m2, size=10),
    np.random.uniform(m2, end, size=10)
))

ax[0].plot(x[(x<=m1) | (x>m2)], np.ones(len(x[(x<=m1) | (x>m2)])), 'bo')
ax[0].plot(x[(x>m1) & (x<=m2)], np.ones(len(x[(x>m1) & (x<=m2)])), 'ro')
ax[1].plot(x[(x<=m1) | (x>m2)], x[(x<=m1) | (x>m2)]**p, 'bo')
ax[1].plot(x[(x>m1) & (x<=m2)], x[(x>m1) & (x<=m2)]**p, 'ro')
ax[1].plot([15, 55], [100, 2600], 'g')
for a in ax:
    a.set_xticks([]) 
    a.set_yticks([])

plt.subplots_adjust(wspace=0, hspace=0)


# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()