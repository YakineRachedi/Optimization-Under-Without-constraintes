# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:17:18 2024

@author: YAKINE RACHEDI
"""

import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1,2,3],[2,2,1],[3,1,3]])

def F(x,lamda): # F R^n+1 à valeurs dans R^n + 1
    l1 = A @ x + lamda * x
    l2 = np.linalg.norm(x)**2 - 1
    return np.concatenate((l1,np.array([l2]))) # np.concatenate prend 2 objets de meme type ici np.array()
VEC_x = np.array([2,3,4])

def DF(x,lamda):
    result = np.zeros((4,4))
    result[:3,:3] = A + lamda * np.eye(3) # varié les 3 lignes et les 3 colones
    result[:3,-1] = x # varié les 3 lignes et la derniere colonne est fixe
    result[-1,:3] = 2 * x # fixe la derniere ligne et varie la colonne
    result[-1,-1] = 0 # mettre dans la derniere case 0
    return result


# méthode de Newton

def Newton_method(v_ini,DF,F,tol = 1e-4,NitMax = 5000):
    # x = (x1,x2,x3) vecteur de R^3
    # lambda est un scalaire
    
    v = v_ini
    k = 0
    x = v[:len(v)-1]
    lamda = v[-1]
    condition = np.linalg.norm(F(x,lamda))
    while(k < NitMax and condition > tol):
        #direction = np.linalg.inv(DF(v[:len(v)-1],v[-1]))@F(v[:len(v)-1],v[-1])
        #v = v - direction
        inv_DF = np.linalg.inv(DF(x,lamda))
        v = v - np.dot(inv_DF,F(x,lamda))
        x = v[:len(v)-1]
        lamda = v[-1]
        condition = np.linalg.norm(F(x,lamda))
        k += 1
    return v,k
eig_val,eig_vect = np.linalg.eig(A)
print(eig_val)
print(eig_vect)
u_newton,k_newton = Newton_method( np.array([2,1,0,1]),DF, F)
print(u_newton)
print(k_newton)