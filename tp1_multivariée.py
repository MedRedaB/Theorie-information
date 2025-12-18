import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
import scipy.linalg as la

n = 10000
mu = np.array([1,2])
R = np.array([[2,1],[1,2]])
L = la.sqrtm(R)
Z = rnd.randn(2,n)
X = (L@Z).T + mu

# Question 2 
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], s=2, alpha=0.3)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Nuage de points des réalisations')
plt.axis('equal')
plt.tight_layout()
plt.savefig('multivariee_gaussienne_points.png', dpi=200)
plt.show()

# Question 3 (lignes de niveau)
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], s=2, alpha=0.3)
theta = np.linspace(0, 2*np.pi, 1000)
unit = np.vstack((np.cos(theta), np.sin(theta)))

for scale in [1, np.sqrt(5), np.sqrt(10)]:
    pts = scale*L@unit + mu.reshape(2,1)
    plt.plot(pts[0], pts[1], linewidth=2)

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Nuage de points et lignes de niveau')
plt.axis('equal')
plt.tight_layout()
plt.savefig('multivariate_gaussienne_avec_contours.png', dpi=200)
plt.show()
