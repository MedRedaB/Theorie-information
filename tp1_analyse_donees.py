import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.patches import Ellipse

donnees = loadmat('pluv.mat')
cles = [k for k in donnees.keys() if not k.startswith('__')]
nom_variable = cles[0]
P = donnees[nom_variable]
Bordeaux, Nantes, Santiago = P[:,0].flatten(), P[:,1].flatten(), P[:,2].flatten()

def statistiques_empiriques(X, Y):
    moyenne = np.mean([X,Y], axis=1)
    covariance = np.cov(X, Y)
    rho = np.corrcoef(X, Y)[0, 1]
    return moyenne, covariance, rho

def trace_nuage_ellipse(X, Y, nom_couple):
    moyenne, covariance, _ = statistiques_empiriques(X, Y)
    plt.figure(figsize=(5,5))
    plt.scatter(X, Y, s=6, alpha=0.5)
    plt.xlabel(nom_couple.split('/')[0])
    plt.ylabel(nom_couple.split('/')[1])
    plt.title(f'Diagramme_de_dispersion_{nom_couple}')
    valeurs, vecteurs = np.linalg.eigh(covariance)
    ordre = valeurs.argsort()[::-1]
    valeurs = valeurs[ordre]
    vecteurs = vecteurs[:, ordre]
    angle = np.degrees(np.arctan2(vecteurs[1,0], vecteurs[0,0]))
    for k in [1,2,3]:
        largeur, hauteur = 2 * np.sqrt(valeurs) * np.sqrt(k)
        ell = Ellipse(xy=moyenne, width=largeur, height=hauteur, angle=angle, fill=False, linewidth=2)
        plt.gca().add_patch(ell)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{nom_couple.replace("/", "_")}.png', dpi=200)
    plt.show()
    return moyenne, covariance

def information_mutuelle(rho):
    return -0.5 * np.log(1 - rho**2 + 1e-15)

moy_bd_na, cov_bd_na, rho_bd_na = statistiques_empiriques(Bordeaux, Nantes)
moy_bd_sa, cov_bd_sa, rho_bd_sa = statistiques_empiriques(Bordeaux, Santiago)
moy_na_sa, cov_na_sa, rho_na_sa = statistiques_empiriques(Nantes, Santiago)

print("Moyennes Bordeaux/Nantes :", moy_bd_na)
print("Covariance Bordeaux/Nantes :\n", cov_bd_na)
print("Rho Bordeaux/Nantes :", rho_bd_na)

trace_nuage_ellipse(Bordeaux, Nantes, "Bordeaux/Nantes")
trace_nuage_ellipse(Bordeaux, Santiago, "Bordeaux/Santiago")
trace_nuage_ellipse(Nantes, Santiago, "Nantes/Santiago")

I_bd_na = information_mutuelle(rho_bd_na)
I_bd_sa = information_mutuelle(rho_bd_sa)
I_na_sa = information_mutuelle(rho_na_sa)

print("Information mutuelle Bordeaux/Nantes :", I_bd_na)
print("Information mutuelle Bordeaux/Santiago :", I_bd_sa)
print("Information mutuelle Nantes/Santiago :", I_na_sa)
