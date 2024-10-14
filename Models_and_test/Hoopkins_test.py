from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from random import sample, uniform

def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H


def hopkins(X):
    d = X.shape[1]  # Nombre de dimensions
    n = len(X)  # Nombre d'échantillons
    m = int(0.3 * n)  # Proportion d'échantillons pour le test
    
    # Initialiser NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    
    # Variables pour stocker les distances
    ujd = []
    wjd = []
    
    # Générer des points aléatoires dans l'espace des données
    for j in range(m):
        # Générer un point aléatoire dans les limites des colonnes (dimensions)
        random_point = [uniform(np.min(X[:,i]), np.max(X[:,i])) for i in range(d)]
        
        # Calculer les distances des points aléatoires
        u_dist, _ = nbrs.kneighbors([random_point], n_neighbors=2, return_distance=True)
        ujd.append(u_dist[0][1])
        
        # Choisir un point réel aléatoire dans X et calculer la distance
        w_dist, _ = nbrs.kneighbors(X[np.random.randint(0, n)].reshape(1, -1), n_neighbors=2, return_distance=True)
        wjd.append(w_dist[0][1])
    
    # Calcul de la statistique de Hopkins
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    
    return H
