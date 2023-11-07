# +
import numpy as np

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


# -

def knn(Z, k=20):
    """Returns the individual knn for a set of representations.
    
    Args:
        Z: set of representations R^{n \times d}
        k: number of neighbours
    
    Return:
        knn: set of nearest neighbours R^{n \times k}
    """
    
    dists = pdist(Z, metric='euclidean')
    dists = squareform(dists)
    knn_index = np.argsort(dists, axis=-1)
    
    # removing self as a NN
    return knn_index[:, 1:k+1] 


def correlation(Z_before, Z_after, k=20):
    assert Z_before.shape[0] == Z_after.shape[0]
    
    knn_before = knn(Z_before, k)
    knn_after = knn(Z_after, k)
    
    corr = []
    for i in range(knn_before.shape[0]):
        overlap = np.intersect1d(knn_before[i], knn_after[i])
        corr.append(len(overlap)/k)
    return corr


def unit_test():
    Z_before = np.random.rand(1000, 13)
    Z_after = np.random.rand(1000, 50)

    print(f"correlation: {np.mean(correlation(Z_before, Z_after))}")
    # Random baseline is (k/n*100)


