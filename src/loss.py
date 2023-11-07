# +
import numpy as np
import torch

from torch import nn


# parts of the code have been adapted from https://github.com/ryanchankh/mcr2/blob/master/loss.py

def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes."""

    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.0
    return labels_onehot


def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.

    Parameters:
        targets (np.ndarray): matrix with one hot labels

    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)

    """

    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.0
    return Pi


class RateDistortion(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01):
        super(RateDistortion, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps

    def rate(self, W, device='cuda:0'):
        """Empirical Discriminative Loss."""

        p, m = W.shape
        I = torch.eye(p).to(device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.0

    def rate_for_mixture(self, W, Pi, device='cuda:0'):
        """Empirical Compressive Loss."""

        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).to(device)
        compress_loss = 0.0
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det * trPi / m
        return compress_loss / 2.0
    
    def rate_for_continuous(self, W, kernel, device='cuda:0'):
        """Empirical Discriminative Loss with a continuous kernel."""

        p, m = W.shape
        I = torch.eye(p).to(device)
        scalar = p / (m * self.eps)
        cov = W.matmul(W.T)
        kernalized_cov = torch.mul(cov, kernel)
        logdet = torch.logdet(I + self.gam1 * scalar * kernalized_cov)
        return logdet / 2.0


class RateDistortionContinuous(RateDistortion):
    """
    Rate distortion loss for deleting continuous
    protected attribute with a kernel function. 
    """
    
    def forward(self, X, X_raw, kernel, device='cuda:0', scale=1.0):
        const = self.rate(X_raw.T, device)
        R_z = self.rate(X.T, device)
        eq_const = torch.abs(R_z - scale * const)
        
        R_z_K = self.rate_for_continuous(X, kernel, device)
        return -R_z_K, eq_const



