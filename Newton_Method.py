# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 19:08:32 2024

@author: Yakine RACHEDI
"""
import numpy as np
import matplotlib.pyplot as plt
def ellipse1(x,y):
    return x**2 + 0.5 * y**2 - 1

def ellipse2(x,y):
    return 0.5*x**2 + y**2 - 1

x = np.linspace(-2,2,51)
y = np.linspace(-2,2,51)

X,Y = np.meshgrid(x,y)

Z1 = ellipse1(X, Y)
Z2 = ellipse2(X, Y)

plt.contour(X,Y,Z1,levels = [0],colors = 'r')
plt.contour(X,Y,Z2,levels = [0],colors = 'b')
plt.grid(True)
plt.title("Graphe des deux éllipses")

# Méthode de Newton pour résoudre le systeme d'équations

def F(x,y):  # systeme d'équations
    l1 = ellipse1(x, y) # ligne 1
    l2 = ellipse2(x, y) # ligne 2
    return np.array([l1,l2])

def GradElips1(x,y):
    return np.array([2 * x, y])
def GradElips2(x,y):
    return np.array([x, 2 * y])

def DF(x,y):
    g1 = GradElips1(x, y)
    g2 = GradElips2(x, y)
    return np.array([g1,g2])

def Newton_method(v_ini,DF,F,tol = 1e-3,IterMax = 1000):
    v = v_ini
    k = 0
    
    if(np.linalg.det(DF(v[0],v[1])) != 0):
        while(k < IterMax and np.linalg.norm(F(v[0],v[1])) > tol):
            inv_F = np.linalg.inv(DF(v[0],v[1]))
            evaluate_F = F(v[0],v[1])
            v = v - inv_F @ evaluate_F  # ATTENTION ICI C'EST PAS UN PRODUIT DE DEUX VECTEURS
            k += 1
    else:
        print("La matrice n'est pas inversible !")
    return v,k

# la méthode est sensible à l'initialisation : faut voir le dessin 
    #et prendre une idée et choisir une initialisation proche des 4 points d'intersections

v0_lower_rignt = np.array([2,-2])
v_newton_LR,k_newton_LR = Newton_method(v0_lower_rignt,DF,F)
print("IterMax Newton LR",k_newton_LR)    


v0_upper_rignt = np.array([2,2])
v_newton_UR,k_newton_UR = Newton_method(v0_upper_rignt,DF,F)
print("IterMax Newton UR",k_newton_UR)  


v0_lower_left = np.array([-1,-2])
v_newton_LL,k_newton_LL = Newton_method(v0_lower_left,DF,F)
print("IterMax Newton LL",k_newton_LL)  
print(v_newton_LL)


v0_upper_left = np.array([-1,1])
v_newton_UL,k_newton_UL = Newton_method(v0_upper_left,DF,F)
print("IterMax Newton UL",k_newton_UL)  
print(v0_upper_left)

plt.scatter(v_newton_LR[0],v_newton_LR[1],marker='*',color ='k',s = 100)
plt.scatter(v_newton_LL[0],v_newton_LL[1],marker='*',color ='k',s = 100)
plt.scatter(v_newton_UR[0],v_newton_UR[1],marker='*',color ='k',s = 100)
plt.scatter(v_newton_UL[0],v_newton_UL[1],marker='*',color ='k',s = 100) 
plt.contour(X,Y,Z1,levels = [0],colors = 'r')
plt.contour(X,Y,Z2,levels = [0],colors = 'b')


# Stocker tous les points d'intersection dans une liste
points_intersection = [v_newton_LR, v_newton_LL, v_newton_UR, v_newton_UL]

# Tracer tous les points d'intersection ensemble
for point in points_intersection:
    plt.scatter(point[0], point[1], marker='*', color='k', s=100)
    
plt.contour(X,Y,Z1,levels = [0],colors = 'r')
plt.contour(X,Y,Z2,levels = [0],colors = 'b')

    
    
    
    
    