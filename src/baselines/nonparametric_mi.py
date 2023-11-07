# part of the code has been adapted from: https://github.com/BiuBiuBiLL/NPEET_LNC


from scipy import stats
import numpy as np
import scipy.spatial as ss
from scipy.special import digamma,gamma
import numpy.random as nr
import random
import matplotlib.pyplot as plt
import re
from scipy.stats.stats import pearsonr
from scipy.stats import pearsonr, entropy

import numpy.linalg as la
from numpy.linalg import eig, inv, norm, det
from scipy import stats
from math import log,pi,hypot,fabs,sqrt
from scipy.special import softmax

from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from scipy.special import digamma
import numpy as np

# +
import numpy as np
import numpy.random as nr
import scipy.spatial as ss
from scipy.special import digamma
from math import log, fabs
import scipy.linalg as la

class MI:
    @staticmethod
    def zip2(*args):
        return [sum(sublist,[]) for sublist in zip(*args)]
    
    @staticmethod
    def avgdigamma(points,dvec):
        N = len(points)
        tree = ss.cKDTree(points)
        avg = 0.
        for i in range(N):
            dist = dvec[i]
            num_points = len(tree.query_ball_point(points[i],dist-1e-15,p=float('inf'))) 
            avg += digamma(num_points)/N
        return avg
    
    @staticmethod
    def entropy(x, k=3, base=np.exp(1), intens=1e-10):
        """ The classic K-L k-nearest neighbor continuous entropy estimator
            x should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
            if x is a one-dimensional scalar and we have four samples
        """
        assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
        d = len(x[0])
        N = len(x)
        x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
        tree = ss.cKDTree(x)
        nn = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in x]
        const = digamma(N) - digamma(k) + d * log(2)
        return (const + d * np.mean(list(map(log, nn)))) / log(base)


    
    @staticmethod
    def pairwise_dist(Z):
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
        return knn_index
    
    @staticmethod 
    def corr(knn_before, knn_after, k=50):
        corrs = []

        for i in range(knn_before.shape[0]):
            overlap = np.intersect1d(knn_before[i], knn_after[i])
            corrs.append(len(overlap)/k)
        return np.mean(corrs)
    
    @staticmethod 
    def alignment(X, k=5):
        num_samples = len(X)
        z_before = X[:, 0].reshape(-1, 1)
        z_after = X[:, 1].reshape(-1, 1)

        knn_before = MI.pairwise_dist(z_before)
        knn_after = MI.pairwise_dist(z_after)

        k = num_samples // 2
        k1 = knn_before[:, 1:k+1]
        k2 = knn_after[:, 1:k+1]
        ret = MI.corr(k1, k2, k)
        return ret
    
    @staticmethod 
    def degree(X, k=5, divergence='l1'):
        num_samples = len(X)
        z_before = X[:, 0].reshape(-1, 1)
        z_after = X[:, 1].reshape(-1, 1)
        
        
        PD_before = squareform(pdist(z_before, metric='euclidean'))
        PD_after = squareform(pdist(z_after, metric='euclidean'))

        knn_before = np.argsort(PD_before)
        knn_after = np.argsort(PD_after)

        knn_indices_after = knn_after[:, 1:k+1]
        knn_indices_before = knn_before[:, 1:k+1]
        
        mask_before = np.zeros_like(PD_before, dtype=np.float64)
        mask_after = np.zeros_like(PD_before, dtype=np.float64)

        np.put_along_axis(mask_after, knn_indices_after, 1, axis=-1)
        adj_matrix_after = mask_after * PD_after
        degree_matrix_after = np.sum(adj_matrix_after, axis=1)
        
        
        np.put_along_axis(mask_before, knn_indices_before, 1, axis=-1)
        adj_matrix_before = mask_before * PD_before
        degree_matrix_before = np.sum(adj_matrix_before, axis=1)
        
        if divergence == 'l1':
            correlation = np.linalg.norm(softmax(degree_matrix_after) - softmax(degree_matrix_before), ord=1)
        elif divergence == 'l2':
            correlation = np.linalg.norm(softmax(degree_matrix_after) - softmax(degree_matrix_before), ord=2)
        elif divergence == 'kl':
            correlation = entropy(softmax(degree_matrix_before), softmax(degree_matrix_after))
        return correlation
    
        

    @staticmethod
    def mi_Kraskov(X, k=5, base=np.exp(1), alpha=0.25, intens=1e-10):
        '''The mutual information estimator by PCA-based local non-uniform correction(LNC)
           ith row of X represents ith dimension of the data, e.g. X = [[1.0,3.0,3.0],[0.1,1.2,5.4]], if X has two dimensions and we have three samples
           alpha is a threshold parameter related to k and d(dimensionality), please refer to our paper for details about this parameter
        '''
        # N is the number of samples
        N = X[0].shape[0]

        x = np.array(X)

        points = np.column_stack(x)

        tree = ss.cKDTree(points)
        dvec = []
        for i in range(len(x)):
            dvec.append([])

        for idx, point in enumerate(points):
            # Find k-nearest neighbors in joint space, p=inf means max norm
            knn = tree.query(point, k + 1, p=float('inf'))
            points_knn = []
            for i in range(len(x)):
                dvec[i].append(float('-inf'))
                points_knn.append([])
            for j in range(k + 1):
                for i in range(len(x)):
                    points_knn[i].append(points[knn[1][j]][i])

            # Find distances to k-nearest neighbors in each marginal space
            for i in range(k + 1):
                for j in range(len(x)):
                    if dvec[j][-1] < fabs(points_knn[j][i] - points_knn[j][0]):
                        dvec[j][-1] = fabs(points_knn[j][i] - points_knn[j][0])

        ret = 0.
        for i in range(len(x)):
            ret -= MI.avgdigamma(x[i], dvec[i])
        ret += digamma(k) - (float(len(x)) - 1.) / float(k) + (float(len(x)) - 1.) * digamma(len(x[0]))
        return ret
# -

