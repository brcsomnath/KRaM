# +
import os
import sys
import pickle

sys.path.append("../")

from copy import deepcopy
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

from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.linear_model import SGDClassifier

from utils import *
from metrics import *
from model import Net
from rlace import solve_adv_game
from inlp import get_debiasing_projection
# -

# ### Data Loading

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

# ### Linear Debiasing

method = 'rlace'


def run_inlp(X, y, X_dev, y_dev, num_iters=25):
    clf = SGDClassifier
    LOSS = "log"
    ALPHA = 1e-5
    TOL = 1e-4
    ITER_NO_CHANGE = 50
    params = {"loss": LOSS, "fit_intercept": True, "max_iter": 3000000, "tol": TOL, "n_iter_no_change": ITER_NO_CHANGE,
              "alpha": ALPHA, "n_jobs": 64}

    input_dim = X_dev.shape[1]

    P_inlp, accs_inlp = get_debiasing_projection(clf, params, num_iters, X_dev.shape[1], True, -1,
                                                                     X, y, X_dev, y_dev, by_class=False,
                                                                     Y_train_main=None, Y_dev_main=None, dropout_rate=0)

    return P_inlp, accs_inlp


if method == 'inlp':
    num_iter = 21
    num_bins = 10
    if dataset_name in ['adult', 'crimes', 'synthetic']:
        bins = np.array(list(range(0, num_bins)))/num_bins
        z_train_q = np.digitize(z_train, bins)
        z_test_q = np.digitize(z_test, bins)
        P_inlp, accs_inlp = run_inlp(x_train, z_train_q, x_test, z_test_q, num_iters=num_iter)
    if dataset_name in ['jigsaw_religion', 'jigsaw_gender', 'jigsaw_religion_openai', 'jigsaw_gender_openai']:
        P_inlp = []
        # iterate for each label
        for i in range(z_test.shape[1]):
            z_train_ = [x[i] for x in z_train]
            z_test_ = [x[i] for x in z_test]
            
            # quantize them
            bins = np.array(list(range(0, num_bins)))/num_bins
            z_train_q = np.digitize(z_train_, bins)
            z_test_q = np.digitize(z_test_, bins)
            P, accs_inlp = run_inlp(x_train, z_train_q, x_test, z_test_q, num_iters=num_iter)
            P_inlp.append(P)
    else:
        P_inlp, accs_inlp = run_inlp(x_train, z_train, x_test, z_test, num_iters=num_iter)
elif method == 'rlace':
    # ranks = [1, 2, 4, 8, 12, 16, 20]
    ranks = [100]
    DEVICE = "cpu"
    num_bins = 10

    Ps_rlace, accs_rlace = {}, {}
    optimizer_class = torch.optim.SGD
    optimizer_params_P = {"lr": 0.005, "weight_decay": 1e-5, "momentum": 0.0}
    optimizer_params_predictor = {"lr": 0.005, "weight_decay": 1e-5, "momentum": 0.0}
    
    for rank in ranks:
        if dataset_name in ['adult', 'crimes', 'synthetic']:
            bins = np.array(list(range(0, num_bins)))/num_bins
            z_train_q = np.digitize(z_train, bins)
            z_test_q = np.digitize(z_test, bins)
            output = solve_adv_game(x_train, z_train_q-1, x_test, z_test_q-1, rank=rank, device=DEVICE, out_iters=50000,
                                optimizer_class=optimizer_class, optimizer_params_P=optimizer_params_P,
                                optimizer_params_predictor=optimizer_params_predictor, epsilon=0.002,
                                batch_size=128, evalaute_every=10000)
            P = output["P"]
            Ps_rlace[rank] = P
            accs_rlace[rank] = output["score"]
        if dataset_name in ['jigsaw_religion', 'jigsaw_gender', 'jigsaw_religion_openai', 'jigsaw_gender_openai']:
            Ps_rlace = []
            # iterate for each label
            for i in range(z_test.shape[1]):
                z_train_ = [x[i] for x in z_train]
                z_test_ = [x[i] for x in z_test]

                # quantize them
                bins = np.array(list(range(0, num_bins)))/num_bins
                z_train_q = np.digitize(z_train_, bins)
                z_test_q = np.digitize(z_test_, bins)
                output = solve_adv_game(x_train, z_train_q-1, x_test, z_test_q-1, rank=rank, device=DEVICE, 
                                        out_iters=10000,
                                    optimizer_class=optimizer_class, optimizer_params_P=optimizer_params_P,
                                    optimizer_params_predictor=optimizer_params_predictor, epsilon=0.002,
                                    batch_size=128, evalaute_every=10000)
                Ps_rlace.append(output["P"])
        else:
            output = solve_adv_game(x_train, z_train, x_test, z_test, rank=rank, device=DEVICE, out_iters=50000,
                                optimizer_class=optimizer_class, optimizer_params_P=optimizer_params_P,
                                optimizer_params_predictor=optimizer_params_predictor, epsilon=0.002,
                                batch_size=128, evalaute_every=10000)
            P = output["P"]
            Ps_rlace[rank] = P
            accs_rlace[rank] = output["score"]

# +
# save stuff
output_path = f"models/{method}"
os.makedirs(output_path, exist_ok=True)

if method == 'inlp':
    with open(f"{output_path}/P_{dataset_name}.pkl", "wb") as f:
        pickle.dump(P_inlp, f)
elif method == 'rlace':
    with open(f"{output_path}/P_{dataset_name}.pkl", "wb") as f:
        pickle.dump(Ps_rlace, f)
# -

# ###  Evaluation

P = P_inlp if method == 'inlp' else Ps_rlace

# +
if dataset_name in ['jigsaw_religion', 'jigsaw_gender', 'jigsaw_religion_openai', 'jigsaw_gender_openai']:
    x_train_debiased = deepcopy(x_train)
    x_test_debiased = deepcopy(x_test)
    
    y_train_debiased = y_train
    y_test_debiased = y_test
    z_train_debiased = z_train
    z_test_debiased = z_test
    for proj in P:
        x_train_debiased = np.transpose(proj.dot(np.transpose(x_train_debiased)))
        x_test_debiased = np.transpose(proj.dot(np.transpose(x_test_debiased)))
else:
    x_train_debiased = np.transpose(P.dot(np.transpose(x_train)))
    x_test_debiased = np.transpose(P.dot(np.transpose(x_test)))
    y_train_debiased = y_train
    y_test_debiased = y_test
    z_train_debiased = z_train
    z_test_debiased = z_test
# -

if dataset_name not in ['glove', 'crimes', 'synthetic']:
    y_hat = classify_predict(x_train_debiased, y_train_debiased, x_test_debiased, y_test_debiased)
    # score = classify(x_train_debiased, y_train_debiased, x_test_debiased, y_test_debiased)
    score = accuracy_score(y_test_debiased, y_hat)
    print(f"Post debiasing classification score (y): {score}")
elif dataset_name in ['glove', 'deepmoji']:
    score = classify(x_train_debiased[y_train_debiased!=2], 
                     y_train_debiased[y_train_debiased!=2], 
                     x_test_debiased[y_test_debiased!=2], 
                     y_test_debiased[y_test_debiased!=2])
    print(f"Post debiasing classification score (y): {score}")
elif dataset_name == 'crimes':
    mse, predictions = evaluate_regression(x_train_debiased, 
                                           y_train_debiased, 
                                           x_test_debiased, 
                                           y_test_debiased)
    print(f"Post debiasing MSE (y): {mse}")

if dataset_name in ['crimes', 'adult', 'synthetic']:
    mse, predictions = evaluate_regression(x_train_debiased, z_train_debiased, x_test_debiased, z_test_debiased)
    print(f"Post debiasing MSE (z): {mse}")
elif dataset_name in ['jigsaw_religion', 'jigsaw_gender', 'jigsaw_religion_openai', 'jigsaw_gender_openai']:
    for i in range(z_test_debiased.shape[1]):
        z_train_ = [x[i] for x in z_train_debiased]
        z_test_ = [x[i] for x in z_test_debiased]
        mse, predictions = evaluate_regression(x_train_debiased, z_train_, x_test_debiased, z_test_)
        print(f"Post-debiasing MSE z- {i}: {mse}")
elif dataset_name in ['celeba', 'deepmoji']:
    score = classify(x_train_debiased, z_train_debiased, x_test_debiased, z_test_debiased)
    print(f"Post-debiasing Classification accuracy (z): {score}")

if dataset_name in ['jigsaw_religion', 'jigsaw_gender', 'jigsaw_religion_openai', 'jigsaw_gender_openai']:
    y_hat = classify_predict(x_train_debiased, y_train_debiased, x_test_debiased, y_test_debiased)
    y_hat_orig = classify_predict(x_train, y_train, x_test, y_test)
    
    for i in range(z_test_debiased.shape[1]):
        print(f"Index: {i}")
        indices = [idx for idx, x in enumerate(z_test[:, i]) if x > 0.]
        ghist = gdp_hist(y_hat_orig[indices], z_test[:, i][indices])
        gkernel = gdp_kernel(y_hat_orig[indices], z_test[:, i][indices])
        print(f"Pre-debiasing scores: GDP-hist: {ghist}, GDP-kernel: {gkernel}")
        
        indices = [idx for idx, x in enumerate(z_test_debiased[:, i]) if x > 0.]
        ghist = gdp_hist(y_hat[indices], z_test_debiased[:, i][indices])
        gkernel = gdp_kernel(y_hat[indices], z_test_debiased[:, i][indices])
        print(f"Post-debiasing scores: GDP-hist: {ghist}, GDP-kernel: {gkernel}")

if dataset_name == 'adult':
    y_hat_orig = classify_predict(x_train, y_train, x_test, y_test)
    ghist = gdp_hist(y_hat_orig, z_test)
    gkernel = gdp_kernel(y_hat_orig, z_test)
    print(f"Pre-debiasing scores: GDP-hist: {ghist}, GDP-kernel: {gkernel}")
    
    y_hat = classify_predict(x_train_debiased, y_train_debiased, x_test_debiased, y_test_debiased)
    ghist = gdp_hist(y_hat, z_test_debiased)
    gkernel = gdp_kernel(y_hat, z_test_debiased)
    print(f"Post-debiasing scores: GDP-hist: {ghist}, GDP-kernel: {gkernel}")
elif dataset_name == 'crimes':
    y_hat_orig = regression_predict(x_train, y_train, x_test, y_test)
    ghist = gdp_hist(y_hat_orig, z_test)
    gkernel = gdp_kernel(y_hat_orig, z_test)
    print(f"Pre-debiasing scores: GDP-hist: {ghist}, GDP-kernel: {gkernel}")
    
    y_hat = regression_predict(x_train_debiased, y_train_debiased, x_test_debiased, y_test_debiased)
    ghist = gdp_hist(y_hat, z_test_debiased)
    gkernel = gdp_kernel(y_hat, z_test_debiased)
    print(f"Post-debiasing scores: GDP-hist: {ghist}, GDP-kernel: {gkernel}")
elif dataset_name in ['celeba', 'deepmoji']:
    # y_hat = classify_predict(x_train_debiased, y_train_debiased, x_test_debiased, y_test_debiased)
    dp = get_demographic_parity(y_hat, z_test_debiased)
    print(f"Post-debiasing scores: DP: {dp}")

# ### Information Alignment

interval = 20
if dataset_name == 'crimes':
    interval = 5

Z_before = x_test
Z_after = x_test_debiased


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
    for k in tqdm(range(20, min(1001, len(x_test)//2+1), interval)):
        k1 = knn_before[:, 1:k+1]
        k2 = knn_after[:, 1:k+1]
        c = corr(k1, k2, k)
        corrs.append(c)
    return corrs


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

rank = np.linalg.matrix_rank(x_test_debiased)




