import torch
import numpy as np

from collections import Counter
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error


def evaluate_regression(x_train, z_train, x_test, z_test, seed=42):
    reg = MLPRegressor(random_state=seed).fit(x_train, z_train)
    z_pred = reg.predict(x_test)
    return mean_squared_error(z_pred, z_test), z_pred


def regression_predict(x_train, z_train, x_test, z_test, seed=42):
    reg = MLPRegressor(random_state=seed).fit(x_train, z_train)
    z_pred = reg.predict(x_test)
    return z_pred


def classify(x_train, y_train, x_test, y_test, seed=42):
    clf = MLPClassifier(random_state=seed, max_iter=20).fit(x_train, y_train)
    return clf.score(x_test, y_test)


def classify_predict(x_train, y_train, x_test, y_test, seed=42):
    clf = MLPClassifier(random_state=seed, max_iter=20).fit(x_train, y_train)
    return clf.predict(x_test)


def encode(arr):
    return torch.tensor(arr.astype(np.float32))


def generate_debiased_embeddings(dataset_loader, net, device='cuda:0'):
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
    Z = []
    for data, y, z in dataset_loader:
        real_data = data.to(device)

        with torch.no_grad():
            output = net(real_data)

        purged_emb = output.detach().cpu().numpy()
        X.extend(purged_emb)
        Y.extend(y.detach().cpu().numpy())
        Z.extend(z.detach().cpu().numpy())
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    return (X, Y, Z)


def get_demographic_parity(y_hat_main, y_protected):
    """
    Computes Demgraphic parity (DP)

    Arguments:
        y_hat_main: predictions for main task 
        y_protected: protected task labels

    Returns:
        dp: Demographic parity across all labels
    """

    all_y = list(Counter(y_hat_main).keys())

    dp = 0
    for y in all_y:
        D_i = []
        for i in range(2):
            used_vals = y_protected == i
            y_hat_label = y_hat_main[used_vals]
            Di_ = len(y_hat_label[y_hat_label == y]) / len(y_hat_label)
            D_i.append(Di_)
        dp += abs(D_i[0] - D_i[1])

    return dp
