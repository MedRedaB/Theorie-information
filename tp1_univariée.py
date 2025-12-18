import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt

n = 10000
mu = 2
sigma = np.sqrt(9)
nbins = 40

X = sigma*rnd.randn(n) + mu

hist, bin_edges = np.histogram(X, bins=nbins, density=True)
delta = bin_edges[1] - bin_edges[0]

plt.figure(figsize=(8,4))
plt.bar(bin_edges[:-1], hist, width=delta, align='edge', alpha=0.6)
xs = np.linspace(bin_edges[0], bin_edges[-1], 400)
pdf = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(xs-mu)**2/(2*sigma**2))
plt.plot(xs, pdf, color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('Densité')
plt.title('Histogramme et densité Gaussienne univariée')
plt.tight_layout()
plt.savefig('univariee_gaussienne.png', dpi=200)
plt.show()

H_discrete = -np.sum(hist*np.log(hist + 1e-12))*delta
H_est = H_discrete + np.log(delta)
H_theor = 0.5*np.log(2*np.pi*np.e*sigma**2)

print('Entropie estimée H_est =', H_est)
print('Entropie théorique H_theor =', H_theor)
