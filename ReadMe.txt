Support functions:

mds(D,k,typ)
	performs MDS on the distance matrix D (numpy array) and k (integer) is the number of resulting dimensions. 
	typ is the label of the dataset.
returns: dataframe with k+1 columns

knn_dist(D, k)
	given a distance matrix D (numpy array) it finds the k (integer) nearest neighbours of each point.
returns sparse numpy array where the non zero elements are the distances to the k clostest neighbours

Isomap(D,knn,d,typ):
	performs isomap on the distance matrix D (np array) with the reduction to d (integer) dimensions by using knn (integer) neighbours
	typ is the label of the data
returns dataframe with d+1 columns

rec_error(D, D_fit, Isomap=0)
	computes the reconstruction error between the distance matrix D (np array) and the reconstructe distance matrix D_fit (np array)
	 by ||K(D)-K(D_fit)||/n if isomap=0 K is the identity and else K is the Isomap kernel
returns error (float)

K(D)
	uses a nxn matrix D (np array) to compute the isomap kernel
returns nxn matrix (np array)

compare_local(D,D_low,k)
	given the original distance matrix and the recostructed one, compare_local checks how many of the k (integer) neighbours of a datapoint in the original
	are still in the neighborhood of the datapoint after dimension reduction (on average)
returns similarity score (float)