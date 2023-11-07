# +
import os
import sys
import pickle

from data_load import read_dataset
from loss import *

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data_utils

import numpy as np
from tqdm import tqdm
from math import pi, sqrt
from copy import deepcopy
from collections import OrderedDict
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.metrics import accuracy_score, mean_squared_error

from utils import *
from metrics import *
from parametric_mi import *
from model import Net, LinearNet
# -

#
# ### Data Loading

dataset_name = 'glove'

x_train, y_train, z_train, x_test, y_test, z_test = read_dataset(dataset_name)

print(f'Train shape: {np.array(x_train).shape}')
print(f'Test shape: {np.array(x_test).shape}')


if dataset_name in ['jigsaw_religion', 
                    'jigsaw_gender', 
                    'adult', 
                    'celeba',
                    'deepmoji', 
                    'jigsaw_religion_openai', 
                    'jigsaw_gender_openai']:
    score = classify(x_train, y_train, x_test, y_test)
    print(f"Pre-debiasing Classification accuracy (y): {score}")
elif dataset_name == 'crimes':
    mse, predictions = evaluate_regression(x_train, y_train, x_test, y_test)
    print(f"Pre-debiasing MSE (y): {mse}")
elif dataset_name in ['glove', 'distribution']:
    score = classify(x_train[y_train!=2], y_train[y_train!=2], x_test[y_test!=2], y_test[y_test!=2])
    print(f"Pre-debiasing Classification accuracy (y): {score}")

if dataset_name in ['crimes', 'adult', 'synthetic']:
    mse, predictions = evaluate_regression(x_train, z_train, x_test, z_test)
    print(f"Pre-debiasing MSE (z): {mse}")
elif dataset_name in ['jigsaw_religion', 'jigsaw_gender', 'jigsaw_religion_openai', 'jigsaw_gender_openai']:
    for i in range(z_test.shape[1]):
        z_train_ = [x[i] for x in z_train]
        z_test_ = [x[i] for x in z_test]
        mse, predictions = evaluate_regression(x_train, z_train_, x_test, z_test_)
        print(f"Pre-debiasing MSE z- {i}: {mse}")
elif dataset_name in ['celeba', 'deepmoji', 'distribution']:
    score = classify(x_train, z_train, x_test, z_test)
    print(f"Pre-debiasing Classification accuracy (z): {score}")

GPU = 1
device = torch.device('cuda:{}'.format(GPU))
device = torch.device('cpu')


# +
batch_size = 200

if dataset_name in ['adult', 'celeba', 'jigsaw_religion_openai', 'jigsaw_gender_openai']:
    batch_size = 400
elif dataset_name in ['deepmoji']:
    batch_size = 1024

train_dataset = data_utils.TensorDataset(encode(x_train), encode(y_train), encode(z_train))
train_dataloader = data_utils.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = data_utils.TensorDataset(encode(x_test), encode(y_test), encode(z_test))
test_dataloader = data_utils.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# +
if dataset_name == 'glove':
    input_size = 300
    embedding_size = 300
    num_layers=4

LN = nn.LayerNorm(embedding_size, elementwise_affine=False)
# -

# ###  Optimizers

gam1, gam2, eps = 1.0, 1.0, 0.5

linear = False


if linear:
    net = LinearNet(num_layers=num_layers, size=embedding_size, input_size=input_size)
else:
    net = Net(num_layers=num_layers, size=embedding_size, input_size=input_size)
net.to(device)

# +
# Arguments for KNIFE
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--dim', type=int, default=128,
                    help='number of dimensions [%(default)d]')
parser.add_argument('--layers', type=int, default=1,
                    help='number of hidden layers [%(default)d]')
parser.add_argument('--tukey_transform', action='store_true', default=False,
                    help='tukey_transform')
parser.add_argument('--no-optimize_mu', dest='optimize_mu', action='store_false',
                    default=True, help='Optimize means of kernel')
parser.add_argument('--cond_modes', type=int,
                    default=128, help='Number of kernel components for conditional estimation')
parser.add_argument('--use_tanh', action='store_true',
                    default=False, help='use tanh()')
parser.add_argument('--init_std', type=float,
                    default=0.01, help='std for initialization')
parser.add_argument('--cov_diagonal', type=str,
                    default='var', help='Diagonal elements of cov matrix different or the same?')
parser.add_argument('--cov_off_diagonal', type=str,
                    default='var', help='Off-diagonal elements of cov matrix zero?')
parser.add_argument('--average', type=str,
                    default='var', help='Weighted average?')
parser.add_argument('--marg_modes', type=int,
                    default=128, help='Kernel components for marginal estimation')
parser.add_argument('--ff_residual_connection', action='store_true',
                    default=False, help='FF Residual Connections')
parser.add_argument('--ff_activation', type=str,
                    default='tanh', help='FF Activation Function')
parser.add_argument('--ff_layer_norm', default=False,
                    action='store_true', help='Use a NormLayer?')

args = parser.parse_args("")

args.ff_layers = args.layers
args.hidden = args.dim

# +
method = 'infonce'

if method == 'knife':
    I_ZX = KNIFE(args, x_train.shape[1], x_train.shape[1])
    I_ZA = KNIFE(args, 1, x_train.shape[1])
elif method == 'mine':
    I_ZX = MINE(args, x_train.shape[1], x_train.shape[1])
    I_ZA = MINE(args, x_train.shape[1], 1)
elif method == 'club':
    I_ZX = CLUB(args, x_train.shape[1], x_train.shape[1])
    I_ZA = CLUB(args, x_train.shape[1], 1)
elif method == 'infonce':
    I_ZX = InfoNCE(args, x_train.shape[1], x_train.shape[1])
    I_ZA = InfoNCE(args, x_train.shape[1], 1)

optimizer = optim.SGD(net.parameters(),
                      lr=0.001,
                      momentum=0.9,
                      weight_decay=5e-4)
# -

# ### Training

# +
print("Starting training ...")

# hyperparam
lambda_ = 1.
sigma_ = 1


iter_num = 0

epochs = 50
if dataset_name == 'glove':
    epochs = 500


itr = tqdm(range(epochs))

for _, epoch in enumerate(itr, 0):
    for step, (batch_embs, batch_y, batch_z) in enumerate(train_dataloader):
        
        X = batch_embs.to(device)
        A = batch_z.reshape(-1, 1).to(device)
        Z = LN(net(X))
        
        loss = -I_ZX(Z, X)[0] + I_ZA(Z, A)[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        itr.set_description(f"Epoch = {epoch} loss = {loss.item():.6f}")
        iter_num += 1
# -

os.makedirs('../models/MI/', exist_ok=True)
torch.save(net, f'../models/MI/{dataset_name}_{method}_net.pb')



# ### Evaluation

x_train_debiased, y_train_debiased, z_train_debiased = generate_debiased_embeddings(train_dataloader, net, device)
x_test_debiased, y_test_debiased, z_test_debiased = generate_debiased_embeddings(test_dataloader, net, device)

if dataset_name in ['glove', 'celeba']:
    score = classify(x_train_debiased[y_train_debiased!=2], 
                     y_train_debiased[y_train_debiased!=2], 
                     x_test_debiased[y_test_debiased!=2], 
                     y_test_debiased[y_test_debiased!=2])
    print(f"Post debiasing classification score (y): {score}")


# ### Information Alignment

def get_representations_from_loader(loader):
    rep = []
    for step, (batch_embs, batch_y, batch_z) in enumerate(loader):
        rep.extend(batch_embs.detach().cpu().numpy())
    return np.array(rep)

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


def corr(knn_before, knn_after, k=20):
    corr = []
    
    for i in range(knn_before.shape[0]):
        overlap = np.intersect1d(knn_before[i], knn_after[i])
        corr.append(len(overlap)/k)
    return np.mean(corr)


def get_corr_scores(knn_before, knn_after):
    corrs = []
    for k in tqdm(range(10, min(1001, len(x_test)//2+1), 20)):
        k1 = knn_before[:, 1:k+1]
        k2 = knn_after[:, 1:k+1]
        c = corr(k1, k2, k)
        corrs.append(c)
    return corrs


Z_before = get_representations_from_loader(test_dataloader)
Z_after = x_test_debiased

knn_before = pairwise_dist(Z_before)
knn_after = pairwise_dist(Z_after)

corrs = get_corr_scores(knn_before, knn_after)

# +
k = (knn_before.shape[0]) // 2
k1 = knn_before[:, 1:k+1]
k2 = knn_after[:, 1:k+1]
print(f"I_(k=50\%) = {corr(k1, k2, k)}")

k = (knn_before.shape[0]) // 10
k1 = knn_before[:, 1:k+1]
k2 = knn_after[:, 1:k+1]
print(f"I_(k=10\%) = {corr(k1, k2, k)}")
# -

rank = np.linalg.matrix_rank(x_test)


