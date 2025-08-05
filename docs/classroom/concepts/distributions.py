from io import StringIO

import numpy as np
import matplotlib.pyplot as plt

def twospirals(n_points, noise=0.7):
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))), 
            np.hstack((np.zeros(n_points), np.ones(n_points))))

# definindo o tamanho da figura
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

for l in range(2):
    for c in range(2):
        ax[l][c].xaxis.set_visible(False)
        ax[l][c].yaxis.set_visible(False)

N = 1000

# geração das amostras 1
mean1 = [-2, -2]           # média da amostra 1
cov1  = [[1, 0], [0, 1]]  # covariância da amostra 1

x1, y1 = np.random.multivariate_normal(mean1, cov1, N).T # gera 500 valores aleatoriamente

# geração das amostras 2 - verde
mean2 = [2, 2]            # média da amostra 2
cov2  = [[1, 0], [0, 1]]  # covariância da amostra 2

x2, y2 = np.random.multivariate_normal(mean2, cov2, N).T

ax[0][0].plot(x1, y1, '.')
ax[0][0].plot(x2, y2, '.')




X, y = twospirals(1000)
ax[1][0].plot(X[y == 0, 0], X[y == 0, 1], '.')
ax[1][0].plot(X[y == 1, 0], X[y == 1, 1], '.')


plt.axis('equal')
plt.subplots_adjust(wspace=0, hspace=0)


# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())