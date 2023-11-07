# parts of the code has been adapted from: https://github.com/brcsomnath/FaRM/

import os
import sys
import pickle

sys.path.append("../")

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
from utils import *
from metrics import *
from model import Net


### Data Loading

dataset_name = 'glove'

x_train, y_train, z_train, x_test, y_test, z_test = read_dataset(dataset_name)

print(f'Train shape: {np.array(x_train).shape}')
print(f'Test shape: {np.array(x_test).shape}')


if dataset_name in ['jigsaw_religion_openai', 'jigsaw_gender_openai']:
    score = classify(x_train, y_train, x_test, y_test)
    print(f"Pre-debiasing Classification accuracy (y): {score}")
elif dataset_name == 'crimes':
    mse, predictions = evaluate_regression(x_train, y_train, x_test, y_test)
    print(f"Pre-debiasing MSE (y): {mse}")
elif dataset_name == 'glove':
    score = classify(x_train[y_train!=2], y_train[y_train!=2], x_test[y_test!=2], y_test[y_test!=2])
    print(f"Pre-debiasing Classification accuracy (y): {score}")

if dataset_name in ['crimes', 'synthetic']:
    mse, predictions = evaluate_regression(x_train, z_train, x_test, z_test)
    print(f"Pre-debiasing MSE (z): {mse}")
elif dataset_name in ['jigsaw_religion_openai', 'jigsaw_gender_openai']:
    for i in range(z_test.shape[1]):
        z_train_ = [x[i] for x in z_train]
        z_test_ = [x[i] for x in z_test]
        mse, predictions = evaluate_regression(x_train, z_train_, x_test, z_test_)
        print(f"Pre-debiasing MSE z- {i}: {mse}")

GPU = 0
device = torch.device('cuda:{}'.format(GPU))


# +
batch_size = 200
if dataset_name == 'glove':
    batch_size = 1024

if dataset_name in ['crimes', 'synthetic']:
    num_bins = 10
    bins = np.array(list(range(0, num_bins)))/num_bins
    z_train_ = np.digitize(z_train, bins)
    z_test_ = np.digitize(z_test, bins)

    train_dataset = data_utils.TensorDataset(encode(x_train), 
                                             encode(y_train),
                                             encode(z_train_),
                                             encode(z_train))
    train_dataloader = data_utils.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

    test_dataset = data_utils.TensorDataset(encode(x_test),
                                            encode(y_test),
                                            encode(z_test_),
                                            encode(z_test))
    test_dataloader = data_utils.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
else:
    train_dataset = data_utils.TensorDataset(encode(x_train), encode(y_train), encode(z_train))
    train_dataloader = data_utils.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = data_utils.TensorDataset(encode(x_test), encode(y_test), encode(z_test))
    test_dataloader = data_utils.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# +
if dataset_name == 'glove':
    input_size = 300
    embedding_size = 300
    num_layers=4
elif dataset_name == 'crimes':
    input_size = 121
    embedding_size = 50
    num_layers=3
elif dataset_name == 'synthetic':
    input_size = 100
    embedding_size = 100
    num_layers=3

LN = nn.LayerNorm(embedding_size, elementwise_affine=False)
# -

# ###  Optimizers

gam1, gam2, eps = 1.0, 1.0, 0.5

net = Net(num_layers=num_layers, size=embedding_size, input_size=input_size)
net.to(device)

# +
criterion = RateDistortionUnconstrained(gam1=gam1, gam2=gam2, eps=eps)

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

iter_num = 0

epochs = 500
if dataset_name in ['crimes']:
    epochs = 100
elif dataset_name == 'glove':
    epochs = 500

itr = tqdm(range(epochs))

for _, epoch in enumerate(itr, 0):
    for step, batch in enumerate(train_dataloader):
        batch_embs, batch_y, batch_z = batch[0], batch[1], batch[2]
        features = LN(net(batch_embs.to(device)))
        labels = batch_z.detach().cpu().numpy()
        labels = labels.astype('int')


        loss = criterion(features, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        itr.set_description(f"Epoch = {epoch} loss = {-loss.item():.6f}")
        iter_num += 1


# +
# os.makedirs('models/farm/', exist_ok=True)
# torch.save(net, f'models/farm/{dataset_name}_net.pb')

# +
# net = torch.load('models/glove_net.pb')
# net.to(device)
# -

### Evaluation

def generate_debiased_quantized(dataset_loader, net, device='cuda:0'):
    """Retrieve debiased embeddings post training.

    Arguments:
        args: arguments
        dataset_loader: pytorch data loader
        net: \phi(x) network
    
    Return:
        dataset: [debiased_embedding, y, z]
    """

    X = []
    Y = []
    Z_q = []
    Z = []
    for data, y, z_q, z in tqdm(dataset_loader):
        real_data = data.to(device)

        with torch.no_grad():
            output = net(real_data)

        purged_emb = output.detach().cpu().numpy()
        X.extend(purged_emb)
        Y.extend(y.detach().cpu().numpy())
        Z.extend(z.detach().cpu().numpy())
        Z_q.extend(z_q.detach().cpu().numpy())
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    Z_q = np.array(Z_q)
    return (X, Y, Z, Z_q)


if dataset_name in ['crimes', 'synthetic']:
    x_train_debiased, y_train_debiased, z_train_debiased, _ = generate_debiased_quantized(
        train_dataloader, net, device)
    x_test_debiased, y_test_debiased, z_test_debiased, _ = generate_debiased_quantized(
        test_dataloader, net, device)
else:
    x_train_debiased, y_train_debiased, z_train_debiased = generate_debiased_embeddings(train_dataloader, net, device)
    x_test_debiased, y_test_debiased, z_test_debiased = generate_debiased_embeddings(test_dataloader, net, device)

if dataset_name not in ['glove', 'crimes', 'synthetic']:
    score = classify(x_train_debiased, y_train_debiased, x_test_debiased, y_test_debiased)
    print(f"Post debiasing classification score: {score}")
elif dataset_name == 'glove':
    score = classify(x_train_debiased[y_train_debiased!=2], 
                     y_train_debiased[y_train_debiased!=2], 
                     x_test_debiased[y_test_debiased!=2], 
                     y_test_debiased[y_test_debiased!=2])
    print(f"Post debiasing classification score: {score}")
elif dataset_name == 'crimes':
    mse, predictions = evaluate_regression(x_train_debiased, 
                                           y_train_debiased, 
                                           x_test_debiased, 
                                           y_test_debiased)
    print(f"Post debiasing MSE (y): {mse}")



if dataset_name in ['crimes', 'synthetic']:
    mse, predictions = evaluate_regression(x_train_debiased, z_train_debiased, x_test_debiased, z_test_debiased)
    print(f"Post debiasing MSE (z): {mse}")
elif dataset_name in ['jigsaw_religion_openai', 'jigsaw_gender_openai']:
    for i in range(z_test_debiased.shape[1]):
        z_train_ = [x[i] for x in z_train_debiased]
        z_test_ = [x[i] for x in z_test_debiased]
        mse, predictions = evaluate_regression(x_train_debiased, z_train_, x_test_debiased, z_test_)
        print(f"Post-debiasing MSE z- {i}: {mse}")
elif dataset_name in ['celeba', 'deepmoji']:
    score = classify(x_train_debiased, z_train_debiased, x_test_debiased, z_test_debiased)
    print(f"Post-debiasing Classification accuracy (z): {score}")


if dataset_name == 'crimes':
    y_hat_orig = regression_predict(x_train, y_train, x_test, y_test)
    ghist = gdp_hist(y_hat_orig, z_test)
    gkernel = gdp_kernel(y_hat_orig, z_test)
    print(f"Pre-debiasing scores: GDP-hist: {ghist}, GDP-kernel: {gkernel}")
    
    y_hat = regression_predict(x_train_debiased, y_train_debiased, x_test_debiased, y_test_debiased)
    ghist = gdp_hist(y_hat, z_test_debiased)
    gkernel = gdp_kernel(y_hat, z_test_debiased)
    print(f"Post-debiasing scores: GDP-hist: {ghist}, GDP-kernel: {gkernel}")

# ### Information Alignment

interval = 20
if dataset_name == 'crimes':
    interval = 5


def get_representations_from_loader(loader):
    rep = []
    for step, batch in enumerate(loader):
        rep.extend(batch[0].detach().cpu().numpy())
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
    for k in tqdm(range(10, min(1001, len(x_test)//2+1), interval)):
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
k = len(x_test) // 2
k1 = knn_before[:, 1:k+1]
k2 = knn_after[:, 1:k+1]
print(f"I_(k=50\%) = {corr(k1, k2, k)}")

k = len(x_test) // 10
k1 = knn_before[:, 1:k+1]
k2 = knn_after[:, 1:k+1]
print(f"I_(k=10\%) = {corr(k1, k2, k)}")
# -
