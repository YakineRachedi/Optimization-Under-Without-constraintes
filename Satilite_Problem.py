# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:45:56 2024

@author: Yakine RACHEDI
"""
import numpy as np
import matplotlib.pyplot as plt

# in meters
#position satellite 1 :
pos1 = [5000000, 3632713, 19021130]
m1 = 3917263658 # pseudo-distance satellite 1
#position satellite 2 :
pos2 = [-5000000, 15388418, 11755705]
m2 = 3917265503# pseudo-distance satellite 2
#position satellite 3 :
pos3 = [11180340, 3632713, -16180340]
m3 = 3917273967# pseudo-distance satellite 3
#position satellite 4 :
pos4 = [9510565, 6909830, 16180339]
m4 = 3917263997# pseudo-distance satellite 4
#definition of arrays Xi, Yi, Zi, mi
Xi = np.array([pos1[0], pos2[0], pos3[0], pos4[0]])
Yi = np.array([pos1[1], pos2[1], pos3[1], pos4[1]])
Zi = np.array([pos1[2], pos2[2], pos3[2], pos4[2]])
mi = np.array([m1, m2, m3, m4])

def distance(x,y,z):
    return np.sqrt((Xi - x)**2 + (Yi - y)**2 + (Zi - z)**2)

def F(v) : # F R^4 à valeurs dans R^4
    x,y,z,w = v
    F = np.zeros(4)
    F = distance(x,y,z) + w - mi
    return(F)

def JF(v) : # F à valeurs vectoriels donc une jacobienne !
    x, y, z, w = v
    J = np.zeros((4, 4))
    # les dérivées par rapport à chaque comosante sont des veteurs colonnes
    J[:,0] = - (Xi - x) / distance(x,y,z)  # on varié la ligne et on fixe la colone 0
    J[:,1] = - (Yi - y) / distance(x,y,z)
    J[:,2] = - (Zi - z) / distance(x,y,z)
    J[:,3] = np.ones(4)
    
    return(J)




def Newton(F, JF, v0, Tol, IterMax):
    v = v0
    it = 0
    S = []
    S.append(v)
    while(it < IterMax and np.linalg.norm(F(v)) > Tol):
        sol = np.linalg.solve(JF(v),F(v))
        v = v - sol
        S.append(v)
        it += 1
    if(it == IterMax or it == 0):
        cvg = False
    else:
        cvg = True
    S = np.array(S)
    return(v, S, cvg, it)

v_newton , vk_newton , cvg_newton, it_newton = Newton(F, JF, np.array([0,0,0,0]), 1e-4, 50)
print("Convergence ?",cvg_newton)
print("Nombre d'itérations ",it_newton)
print("valeur de la solution",v_newton)