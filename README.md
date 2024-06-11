# Verson opérationnelle
-> AudienceClustering_K-prototype<br>


# Audience Clustering Beta
 Beta version of audience clustering.<br>
 L'outil Audience classification est un système puissant qui permet aux entreprises de comprendre et de catégoriser leurs clients en fonction de différents critères. 

 Cela leur permet de personnaliser leurs interactions et leurs stratégies pour maximiser la satisfaction des clients et les performances commerciales.
 
## Prétraitement
1. Hopkins Statistics : <br>
Pour comprendre si l'ensemble de données peut être regroupé, nous avons utilisé la statistique de Hopkins, qui teste l'aléa spatial des données et indique la tendance de regroupement ou la capacité des données à être regroupées. Elle calcule la probabilité qu'un ensemble de données donné soit généré par une distribution uniforme (Alboukadel Kassambara, s.d.). L'interprétation est la suivante pour un ensemble de données de dimensions 'd' : <br>
<ul> 
<li>Si la valeur est proche de 0,5 ou inférieure, les données sont uniformément réparties et il est donc peu probable qu'il y ait des clusters statistiquement significatifs. </li> <br>
<li>Si la valeur se situe entre {0,7,...,0,99}, cela indique une forte tendance au regroupement et donc une probabilité élevée d'avoir des clusters statistiquement significatifs.</li> <br> <br>
</ul>
 Pour se faire on utilise la fonction qui calcule le test en selection uniquement <strong>les colonnes de types int</strong>

```python
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

# Lancement
hopkins(df[Col_int])
```

 
## Étapes principales du Clustering :

1. Sélection des données :<br>
La première étape consiste à choisir les données pertinentes pour l'analyse. Ces données peuvent être des vecteurs numériques représentant les caractéristiques des éléments à regrouper.

2. Choix du nombre de clusters :<br>
Avant de procéder au clustering, il est essentiel de déterminer le nombre de clusters que l'on souhaite obtenir. Ce choix peut être basé sur des connaissances préalables du domaine ou sur des techniques d'évaluation telles que l'analyse de la silhouette ou la méthode du coude.<br>
On se base sur le modèle de Kmeans et on texte sur un échantillon le nombre de cluster qui donnes un score de silhouette le plus grand
```python
Echantillon =Echantillon[Col_int]
L=[]
for n_clusters in range(2,20):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(Echantillon)
    clusters = kmeans.predict(Echantillon)
    silhouette_avg = silhouette_score(Echantillon, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    L.append([n_clusters,round(silhouette_avg,2)])
```
3. Choix de l'algorithme de clustering :<br>
Il existe de nombreux algorithmes de clustering tels que le K-Means, le clustering hiérarchique agglomératif, le DBSCAN (Density-Based Spatial Clustering of Applications with Noise), le Mean Shift, etc. Chaque algorithme a ses propres forces et faiblesses, et le choix dépend souvent du type de données et des objectifs du clustering.

```python
# Instanciation du modèle de clustering k-prototypes
kproto = KPrototypes(n_clusters=K, init='Cao', random_state=42)

# Entraînement du modèle
clusters = kproto.fit_predict(matrix_for_cluster.values, categorical=[2,3,4])

# Ajout des clusters attribués aux données
data['Cluster'] = clusters

```
