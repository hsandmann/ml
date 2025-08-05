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
fig, ax = plt.subplots(2, 2, figsize=(12, 12))

for l in range(2):
    for c in range(2):
        ax[l][c].xaxis.set_visible(False)
        ax[l][c].yaxis.set_visible(False)
        ax[l][c].set_aspect('equal', adjustable='box')
        ax[l][c].set_xlim(-7, 7)
        ax[l][c].set_ylim(-7, 7)

N = 1000

#####################
x1, y1 = np.random.multivariate_normal(
    [-2.5, -2.5],
     [[1, 0], [0, 1]],
    N
).T

x2, y2 = np.random.multivariate_normal(
    [2.5, 2.5],
    [[1, 0], [0, 1]],
    N
).T

ax[0][0].plot(
    x1, y1, '.',
    x2, y2, '.'
)
#####################
x3, y3 = np.random.multivariate_normal(
    [-2.5, 2.5],
     [[1, 0], [0, 1]],
    N
).T

x4, y4 = np.random.multivariate_normal(
    [2.5, -2.5],
    [[1, 0], [0, 1]],
    N
).T

ax[0][1].plot(
    np.hstack((x1, x2)), np.hstack((y1, y2)), '.',
    np.hstack((x3, x4)), np.hstack((y3, y4)), '.'
)
#####################
xc, yc = np.random.multivariate_normal(
    [0, 0],
    [[1, 0], [0, 1]],
    N
).T

noise = .5
radius = 2.5
theta = np.linspace(0, 2 * np.pi, N)

xt = radius * np.cos(theta)
yt = radius * np.sin(theta)

xt = xt + np.random.normal(xt, noise)
yt = yt + np.random.normal(yt, noise)

# Plot the surface
ax[1][0].plot(
    xc, yc, '.',
    xt, yt, '.'
)

#####################
X, y = twospirals(N)
ax[1][1].plot(X[y == 0, 0], X[y == 0, 1], '.')
ax[1][1].plot(X[y == 1, 0], X[y == 1, 1], '.')

#####################
plt.axis('equal')
plt.subplots_adjust(wspace=0, hspace=0)

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())