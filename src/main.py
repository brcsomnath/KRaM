"""
Implementation of Kernelized Rate Distortion Maximizer (KRaM).
"""


from loss import *

import wandb
import argparse

import torch
from torch import nn, optim
import torch.utils.data as data_utils

import numpy as np
from tqdm import tqdm

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


from utils import *
from metrics import *
from model import Net, LinearNet
from data_load import read_dataset

# +

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', 
                    type=str,
                    default='glove',
                    help='Name of dataset')
parser.add_argument('--batch_size',
                    type=int,
                    default=200,
                    help='Batch size for training')
parser.add_argument('--num_epochs', 
                    type=int,
                    default=10,
                    help='Number of epochs for training')
parser.add_argument('--gpu', 
                    type=int,
                    default=0,
                    help='GPU Id')
parser.add_argument('--linear', 
                    type=bool,
                    default=False,
                    help='Whether the network is linear')
parser.add_argument('--learning_rate', 
                    type=float,
                    default=0.001,
                    help='learning rate of training')
parser.add_argument('--momentum', 
                    type=float,
                    default=0.9,
                    help='momentum of SGD')
parser.add_argument('--weight_decay', 
                    type=float,
                    default=5e-4,
                    help='weight decay of SGD')
parser.add_argument('--lambda_', 
                    type=float,
                    default=1,
                    help='lambda - threshold weight')
parser.add_argument('--sigma', 
                    type=float,
                    default=0.01,
                    help='parameter of the kernel function')
parser.add_argument('--gam1', 
                    type=float,
                    default=1.,
                    help='gamma 1')
parser.add_argument('--gam2', 
                    type=float,
                    default=1.,
                    help='gamma 2')
parser.add_argument('--eps', 
                    type=float,
                    default=0.5,
                    help='epsilon')
parser.add_argument('--run_id', 
                    type=int,
                    default=1,
                    help='Run ID.')
parser.add_argument('--info_alignment_switch', 
                    type=bool,
                    default=False,
                    help='do we compute information alignment?')
args = parser.parse_args()

# Define a dictionary of configuration settings
config = {
    'dataset_name': args.dataset_name,
    'batch_size': args.batch_size,
    'num_epochs': args.num_epochs,
    'linear': args.linear,
    'learning_rate': args.learning_rate,
    'momentum': args.momentum,
    'weight_decay': args.weight_decay,
    'lambda_': args.lambda_,
    'sigma': args.sigma,
    'gamma1': args.gam1,
    'gamma2': args.gam2,
    'eps': args.eps,
    'num_layers': 1,
}


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

def get_representations_from_loader(loader):
    rep = []
    for step, (batch_embs, batch_y, batch_z) in enumerate(loader):
        rep.extend(batch_embs.detach().cpu().numpy())
    return np.array(rep)


def initial_scores(dataset_name):
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


def get_network_params(dataset_name):
    if dataset_name in ['jigsaw_religion_openai', 'jigsaw_gender_openai']:
        input_size = 768
        embedding_size = 768
        num_layers=3
    elif dataset_name == 'glove':
        input_size = 300
        embedding_size = 300
        num_layers=4
    elif dataset_name == 'crimes':
        input_size = 121
        embedding_size = 121
        num_layers=3
    elif dataset_name == 'synthetic':
        input_size = 100
        embedding_size = 100
        num_layers=3
    return input_size, embedding_size, num_layers


def train(config=None):
    with wandb.init(project="KRaM", 
                    config=config,
                    settings=wandb.Settings(program_relpath="main.py", 
                    disable_git=True, save_code=False)):
        config = wandb.config

        dataset_name = config.dataset_name
        x_train, y_train, z_train, x_test, y_test, z_test = read_dataset(dataset_name)

        device = torch.device('cuda:{}'.format(args.gpu))

        train_dataset = data_utils.TensorDataset(encode(x_train), encode(y_train), encode(z_train))
        train_dataloader = data_utils.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

        test_dataset = data_utils.TensorDataset(encode(x_test), encode(y_test), encode(z_test))
        test_dataloader = data_utils.DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

        if args.info_alignment_switch:
            Z_before = get_representations_from_loader(test_dataloader)
            knn_before = pairwise_dist(Z_before)
            k50, k10 = len(x_test) // 2, len(x_test) // 10
            k1_50 = knn_before[:, 1:k50+1]
            k1_10 = knn_before[:, 1:k10+1]

        input_size, embedding_size, num_layers = get_network_params(config.dataset_name)
        LN = nn.LayerNorm(embedding_size, elementwise_affine=False)


        if config.linear:
            net = LinearNet(num_layers=config.num_layers,
            size=embedding_size, input_size=input_size)
        else:
            net = Net(num_layers=config.num_layers,
            size=embedding_size, input_size=input_size)
        net.to(device)

        criterion = RateDistortionContinuous(gam1=config.gam1, 
                                            gam2=config.gam2, 
                                            eps=config.eps)
        optimizer = optim.SGD(net.parameters(),
                            lr=config.learning_rate,
                            momentum=config.momentum,
                            weight_decay=config.weight_decay)



        print("Starting training ...")
        epochs = config.num_epochs + 1

        itr = tqdm(range(epochs))
        num_iter = 0

        for _, epoch in enumerate(itr, 0):
            for step, (batch_embs, batch_y, batch_z) in enumerate(train_dataloader):

                X_orig = batch_embs.to(device)
                if dataset_name in ['jigsaw_religion_openai', 'jigsaw_gender_openai']:
                    normalized_z = batch_z/ batch_z.norm(dim=1)[:, None]
                    kernel = gaussian_kernel_multi(normalized_z, sigma=config.sigma).to(device)
                elif dataset_name == 'glove':
                    kernel = gaussian_kernel(batch_z, sigma=config.sigma, data_type='categorical').to(device)
                else:
                    kernel = gaussian_kernel(batch_z, sigma=config.sigma).to(device)

                Rzk, eq = criterion(LN(net(X_orig)), 
                                    X_orig, 
                                    kernel=kernel, 
                                    device=device,
                                    scale=config.scale)

                loss = Rzk + config.lambda_ * eq
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_iter += 1

                itr.set_description(f"Epoch = {epoch} Rzk = {-Rzk.item():.6f} eq: {eq.item():.6f}")
            
            if epoch % 100 == 0:
                x_train_debiased, y_train_debiased, z_train_debiased = generate_debiased_embeddings(train_dataloader, net, device)
                x_test_debiased, y_test_debiased, z_test_debiased = generate_debiased_embeddings(test_dataloader, net, device)

                gkernel = 0.
                if dataset_name == 'crimes':
                    y_score, y_test_pred = evaluate_regression(x_train_debiased, 
                                                         y_train_debiased, 
                                                         x_test_debiased, 
                                                         y_test_debiased)
                    gkernel = gdp_kernel(y_test_pred, z_test_debiased)

                if dataset_name in ['crimes', 'synthetic']:
                    mse, predictions = evaluate_regression(x_train_debiased, 
                                                           z_train_debiased,
                                                           x_test_debiased,
                                                           z_test_debiased)
                elif dataset_name == 'glove':
                    y_score = classify(x_train_debiased[y_train_debiased!=2], 
                                        y_train_debiased[y_train_debiased!=2], 
                                        x_test_debiased[y_test_debiased!=2], 
                                        y_test_debiased[y_test_debiased!=2])
                
                
                Ik_50, Ik_10 = 0.0, 0.0
                if args.info_alignment_switch:
                    Z_after = x_test_debiased
                    knn_after = pairwise_dist(Z_after)
                    k2 = knn_after[:, 1:k50+1]
                    Ik_50 = corr(k1_50, k2, k50)

                    k2 = knn_after[:, 1:k10+1]
                    Ik_10 = corr(k1_10, k2, k10)
                
                if dataset_name in ['crimes']:
                    wandb.log({"loss": gkernel, 
                            "epoch": epoch,
                            "score_y": y_score,
                            "mse_z": mse,
                            "Ik_50": Ik_50,
                            "Ik_10": Ik_10})
                elif dataset_name == 'synthetic':
                    wandb.log({
                            "epoch": epoch,
                            "mse_z": mse,
                            "Ik_50": Ik_50,
                            "Ik_10": Ik_10})
                elif dataset_name == 'glove':
                    wandb.log({
                            "epoch": epoch,
                            "score_y": y_score,
                            "Ik_50": Ik_50,
                            "Ik_10": Ik_10})

# +
sweep_config = {
    'project': f'kram_{args.dataset_name}_{args.run_id}',
    'method': 'grid',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'   
    },
    'parameters': {
        'sigma': {'values': [args.sigma]}, 
        'lambda_': {'values': [args.lambda_]},
        'num_epochs': {'values': [args.num_epochs]},
        'dataset_name': {'values': [args.dataset_name]},
        'batch_size': {'values': [args.batch_size]},
        'linear': {'values': [args.linear]},
        'learning_rate': {'values': [args.learning_rate]},
        'momentum': {'values': [args.momentum]},
        'weight_decay': {'values': [args.weight_decay]},
        'gam1': {'values': [args.gam1]},
        'gam2': {'values': [args.gam2]},
        'eps': {'values': [args.eps]},
        'num_layers': {'values': [3]},
        'scale': {'values': [1]},
    }
}


sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, train)
# -




