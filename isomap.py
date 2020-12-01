import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def knn_dist(d_matrix, k):
    D_knn = np.zeros(d_matrix.shape)
    #get the indices of the k lowest values and set these indices in D_knn to the same values as in D
    #the rest stays 0
    for i, row in enumerate(d_matrix):
        index = row.argsort()[:k]
        D_knn[i][index] = row[index]
        
    return D_knn

# create 0,1 graph from k_nearest neighbour matrix
def create_graph(knn_matrix):
    graph = knn_matrix > 0
    graph = graph*1
    return graph

#compute the k_nearest neighbour matrix with the new connections
def tuned_knn(d_matrix, n_components, labels, knn_d):
    #get individual combinations
    comb = [(i,j) for i in range(n_components)  for j in range(i,n_components) if i != j]
    tuned_knn = np.copy(knn_d)
    dist = []
    for c in comb:
        dist.append(component_dist(labels, d_matrix, c[0], c[1]))
    dist = sorted(dist, key=lambda x: x[0])
    for i in range(n_components-1):
        l,j = dist[i][1]
        tuned_knn[l,j] = dist[i][0]
    
    return tuned_knn

#calculate the shortest distance between the components c1 and c2
def component_dist(labels, d_matrix, c1, c2):
    l1 = [i for i,j in enumerate(labels) if j==c1]
    l2 = [i for i,j in enumerate(labels) if j==c2]
    n,n = d_matrix.shape
    temp_d = d_matrix + np.eye(n)*10**20 #avoid that the diagonal is measured as shortest distance
    dist = 100000
    lab = 0
    for i in l1:
        temp_dist = min(temp_d[i][l2])
        ind = np.argmin(temp_d[i][l2])
        if temp_dist < dist:
            dist = temp_dist
            lab = [i,l2[ind]]

    return dist, lab

#check for components in the given graph according to the k_nearest neighbour matrix
def check_components(knn_matrix):
    graph = create_graph(knn_matrix)
    graph = csr_matrix(graph)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    return n_components, labels
