# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 14:52:30 2024

@author: YAKINE RACHEDI
"""

import numpy as np
import matplotlib.pyplot as plt

n = 10
h = 1 / (n+1)
#b = np.ones(n)
b = 3 * np.ones(n) - 6 * h * np.arange(0,n,1)  # deuxieme b
A = 2 * np.diag(np.ones(n)) - np.diag(np.ones(n-1),1) - np.diag(np.ones(n-1),-1)
"""
A = np.zeros((n,n))
for i in range(n-1):
    for j in range(n-1):
        A[i,i] = 2
        A[i,i+1] = -1
        A[i+1,i] = -1
    A[n-1,n-1] = 2
print(A)
"""

def GradJ(x):
    return (1 / h) * A@x - h * b

# solution de GradJ = 0
sol = np.linalg.solve((1/h) * A, h* b)
print(sol)

h_list = []
for i in range(n+2):  # n+2 pour prise en compte le n+1 !
    h_list.append(i * h)

sol = np.append([0],sol) # mettre les conditions aux limites
sol = np.append(sol,[0])

plt.plot(h_list,sol)
plt.title("La solution u bar du probleme de minimisation")
plt.xlabel("valeurs de h")
plt.ylabel("Solution")
plt.grid(True)
plt.show()


# de meme 
h_list = [(i * h) for i in range(n+2)]
plt.plot(h_list,sol)
plt.title("La solution u bar du probleme de minimisation")
plt.xlabel("valeurs de h")
plt.ylabel("Solution")
plt.grid(True)
plt.show()


def gfstp(x0,GradJ,rho,tol = 1e-4,IterMax = 1000):
    x = x0
    k = 0
    Suite_val = []
    Suite_val.append(x)
    while(k < IterMax and np.linalg.norm(GradJ(x)) > tol):
        x = x - rho * GradJ(x)
        k += 1
        Suite_val.append(x)
    if(k == IterMax or k == 0):
        convergence = False
    else:
        convergence = True
    Suite_val = np.array(Suite_val)
    return Suite_val,x,k,convergence


#rho = np.arange(0.01,0.07,0.001)
rho = np.linspace(0.01,0.07,101)
nb_Iter1 = np.zeros(len(rho))


x_ini1 = np.zeros(n)

for i in range(len(rho)):
    uk,u_bar,k,conv = gfstp(x_ini1,GradJ,rho[i])
    nb_Iter1[i] = k
    
plt.plot(rho,nb_Iter1)
plt.title("Nombre d'itérations en fonction du pas")
plt.xlabel("valeurs du pas")
plt.ylabel("nombre d'itérations")
plt.grid(True)
plt.show()
    
index = np.argmin(nb_Iter1)
print("Le meilleur rho pour cette initialisation est : ", rho[index])

nb_Iter2 = np.zeros(len(rho))
x_ini2 = np.full(n,fill_value=1)  # ça dépend de l'initialisation !
for i in range(len(rho)):
    uk,u_bar,k,conv = gfstp(x_ini2,GradJ,rho[i])
    nb_Iter2[i] = k

plt.plot(rho,nb_Iter2)
plt.title("Nombre d'itérations en fonction du pas avec une initialisation différente")
plt.xlabel("valeurs du pas")
plt.ylabel("nombre d'itérations")
plt.grid(True)
plt.show()    
index = np.argmin(nb_Iter1)
print("Le meilleur rho pour cette initialisation est : ", rho[index])
print("On remarque que rho ne dépend pas de l'initialisation !")
plt.scatter(uk[:,0],uk[:,1])
plt.title("comportement de la suite à chaque iteration")
plt.grid(True)
# cherchons rho qui conduit au petit nombre d'itérations, autrement dit le rho optimal
# on trouve une constante de Lipsichtz M = norm(A) / h
# apres un calcul sur feuil on trouve d'apres le résultat de cours
# rho_opt = norm(gradient ^2) / gradient * hessienne * gradient

# Gradient à pas optimal

def gopt(GradJ,x0,tol = 1e-3,IterMax = 1000):
    x = x0
    k = 0
    Suite_val = []
    Suite_val.append(x)
    HessienneJ = (1/h) * A # calcul à la main
    while(k < IterMax and np.linalg.norm(GradJ(x)) > tol):
        # étape précédente xk
        norm_grad = np.linalg.norm(GradJ(x))
        rho_opt = norm_grad ** 2 / ((GradJ(x) @ HessienneJ) @ GradJ(x))
        # étape suivante xk+1
        x = x - rho_opt * GradJ(x)
        k += 1
        Suite_val.append(x)
    if(k == IterMax or k == 0):
        convergence = False
    else:
        convergence = True
    Suite_val = np.array(Suite_val)
    return Suite_val,x,k,convergence

uk_opt,u_opt,k_opt,cv_opt = gopt(GradJ,x_ini1)
print("Convergence optimale ?",cv_opt)
print("Iteration Max",k_opt)
 
#Recap : 
#quand n augmente l'erreur augmente
#quand itmax augmente l'erreur diminue
#et quand tol augmente, l'erreur augmente

# erreur 
_,u_best,_,_ = gfstp(x_ini2,GradJ,rho[index])
sol = sol[1:-1] # enleve les cond aux lims = 0
print("L'erreur en norme L2 de l'erreur : ", np.linalg.norm(sol - u_best) )
    
