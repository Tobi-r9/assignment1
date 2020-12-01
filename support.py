import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils.graph import graph_shortest_path
import cmath
from sklearn import svm
from sklearn.metrics import accuracy_score 

def do_pca(data, components, typ):
    pca = PCA(n_components = components)
    pca.fit(data)
    X = pca.transform(data)
    df_pca = pd.DataFrame(data=X, columns=['comp'+str(i) for i in range(components)])
    df_pca['type'] = typ
    return df_pca

def mds(D,k,typ): #k is the dimension to which we reduce the data
    #compute S
    I = np.ones(D.shape)
    n = D.shape[0]
    S = -1/2 * (D-(1/n)*D@I - (1/n)*I@D + (1/n**2)*I@D@I)

    #eigenvalue decomposition and computation of the k-dimensional representation
    eigenValues, eigenVectors = np.linalg.eig(S)
    mds = np.eye(k,n)@ np.sqrt(np.diag(abs(eigenValues))) @eigenVectors.T
    
    #make complex data real
    h,p = mds.shape
    temp = np.zeros((h,p))
    for i in range(h):
        for j in range(p):
            temp[i,j] = mds[i,j].real

    #create dataframe for easy handling
    df_mds = pd.DataFrame(data=temp.T, columns=["comp"+str(i) for i in range(len(mds))])
    df_mds['type'] = typ

    return df_mds


def knn_dist(D, k): 
    #k number of neighbours
    D_knn = np.zeros(D.shape)
    #get the indices of the k lowest values and set these indices in D_knn to the same values as in D the rest stays 0
    for i, row in enumerate(D):
        index = row.argsort()[:k]
        D_knn[i][index] = row[index]
        
    return D_knn

def Isomap(D,knn,d,typ):
    D_knn = knn_dist(D,knn)
    #computes the shortes path between all points given the Graph matrix D_knn
    graph = graph_shortest_path(D_knn, directed=False)

    return mds(graph,d,typ)
# compute the reconstruction error of the distance matrix
def rec_error(D, D_fit, Isomap=0):
    n,_ = D.shape
    if Isomap:
        KD = K(D)
        KD_fit = K(D_fit)
    else:
        KD = np.copy(D)
        KD_fit = np.copy(D_fit)
    dist = KD-KD_fit
    error = np.linalg.norm(dist, ord='fro')/n
    return error

#Isomap kernel
def K(D):
    n = D.shape[0]
    out = -0.5 * (np.eye(n) - 1/n) * D**2 * (np.eye(n) - 1/n)
    return out

# compute the average fraction of same k nearest neighbours of the data and the lower embedding
def compare_local(D,D_low,k):
    knn1 = knn_dist(D, k)
    knn2 = knn_dist(D_low, k)
    n,_ = D.shape
    C = []
    for i in range(n):
        org = np.nonzero(knn1[i])[0]
        low = np.nonzero(knn2[i])[0]
        cnt = 0
        for i in org:
            if i in low:
                cnt += 1
        C.append(cnt/k) 
    erg = sum(C)/n
    return erg

# do svm on the data and return score to check the seperability
def seperability(C, data):
    temp_df = data.copy()
    y = temp_df['type']
    del temp_df['type']
    clf = svm.SVC(kernel='linear',C=C)
    clf.fit(temp_df, y)
    y_pred = clf.predict(temp_df)
    score = accuracy_score(y,y_pred)
    return score