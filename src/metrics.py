"""Generalized Demographic Parity."""

import torch
import numpy as np

from math import pi, sqrt

def gaussian_kernel(tensor, sigma=0.1, data_type='continuous'):
    protected_var = tensor.reshape((-1, 1))
    pairwise_dist = torch.cdist(protected_var, protected_var)
    kernel = torch.exp(- pairwise_dist / sigma)
    if data_type == 'categorical':
        kernel[kernel<1.] = 0.
    return kernel


def laplace_kernel(tensor, sigma=0.1, data_type='continuous'):
    protected_var = tensor.reshape((-1, 1))
    pairwise_dist = torch.cdist(protected_var, protected_var, p=1)
    kernel = torch.exp(- pairwise_dist / sigma)
    if data_type == 'categorical':
        kernel[kernel<1.] = 0.
    return kernel


def cauchy_kernel(tensor, sigma=0.1, data_type='continuous'):
    protected_var = tensor.reshape((-1, 1))
    pairwise_dist = torch.cdist(protected_var, protected_var, p=1)
    kernel = torch.exp(1/(1 + pairwise_dist / sigma**2))
    if data_type == 'categorical':
        kernel[kernel<1.] = 0.
    return kernel


def gaussian_kernel_multi(tensor, sigma=0.1):
    # cosine distance using normalized vectors
    pairwise_dist = 1 - torch.mm(tensor, tensor.transpose(0,1))
    return torch.exp(- pairwise_dist / sigma)


def gdp_hist(y_hat, s):
    N = y_hat.shape[0]
    m_avg = np.mean(y_hat)
    bandwidth = N ** (-1./3)
    n_bins = int(1./bandwidth)
    
    bins = np.linspace(0, 1, n_bins)
    s_q = np.digitize(s, bins)
    
    
    GDP = 0
    
    for i in range(n_bins):
        bin_elements = y_hat[s_q == i]
        if len(bin_elements) == 0:
            continue
            
        m_s = bin_elements.mean()
        P_s = len(bin_elements) / N
        GDP += abs(m_s - m_avg) * P_s
        
    return GDP


def gdp_kernel(y_hat, s):
    N = y_hat.shape[0]
    m_avg = np.mean(y_hat)
    bandwidth = (N) ** (-1./5)
    n_probes = int(1./bandwidth)
    
    probes = np.linspace(0, 1, n_probes)
    probes = np.repeat(probes, N)
    probes = np.reshape(probes, (-1, N))
    
    data = np.reshape(s, (-1, N))
    kernel_values = np.exp(-((probes - data) ** 2 / (bandwidth ** 2) / 2)) / (sqrt(2 * pi) * bandwidth)
    pdf_values = kernel_values.mean(axis=-1) 
    
    
    m_s = (kernel_values * y_hat).mean(axis=-1) / kernel_values.mean(axis=-1)

    # normalize the PDF distribution
    pdf_values = pdf_values / sum(pdf_values)
    return sum(abs(m_s - m_avg) * pdf_values)
