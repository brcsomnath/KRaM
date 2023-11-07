# +
import os
import sys

sys.path.append("baselines/")

import openai
import math
import pickle
import torch
import urllib
import random
import sklearn
import os.path
import numpy as np
import pandas as pd

import sklearn.preprocessing as preprocessing

from PIL import Image
from tqdm import tqdm
from sklearn.utils import shuffle
from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
# -

dirname = '../data/'


def read_dataset(name, 
                 label=None, 
                 sensitive_attribute=None, 
                 fold=None):
    '''Utility function to load any dataset.'''

    if name == 'crimes':
        y_name = label if label is not None else 'ViolentCrimesPerPop'
        z_name = sensitive_attribute if sensitive_attribute is not None else 'racepctblack'
        fold_id = fold if fold is not None else 1
        return read_crimes(os.path.join(dirname, 'crimes'), 
                           label=y_name,
                           sensitive_attribute=z_name, 
                           fold=fold_id)
    elif name == 'jigsaw_religion_openai':
        return read_jigsaw_openai(dirname, protected_label='religion')
    elif name == 'jigsaw_gender_openai':
        return read_jigsaw_openai(dirname, protected_label='gender')
    elif name == 'glove':
        return load_glove(os.path.join(dirname, 'glove'))
    elif name == 'synthetic':
        return load_synthetic(os.path.join(dirname, 'synthetic'))
    elif name == 'deepmoji':
        return load_deepmoji(os.path.join(dirname, 'deepmoji'))
    else:
        raise NotImplemented('Dataset {} does not exists'.format(name))


# ## DeepMoji

def load_data(path, size, ratio=0.5):
    fnames = ["neg_neg.npy", "neg_pos.npy", "pos_neg.npy", "pos_pos.npy"]
    protected_labels = [0, 1, 0, 1]
    main_labels = [0, 0, 1, 1]
    X, Y_p, Y_m = [], [], []
    n1 = int(size * ratio / 2)
    n2 = int(size * (1 - ratio) / 2)

    for fname, p_label, m_label, n in zip(fnames, protected_labels,
                                          main_labels, [n1, n2, n2, n1]):
        data = np.load(path + "/" + fname)[:n]
        for x in data:
            X.append(x)
        for _ in data:
            Y_p.append(p_label)
        for _ in data:
            Y_m.append(m_label)

    Y_p = np.array(Y_p)
    Y_m = np.array(Y_m)
    X = np.array(X)
    X, Y_p, Y_m = shuffle(X, Y_p, Y_m, random_state=0)
    return X, Y_p, Y_m


def load_deepmoji(path="../data/deepmoji/", ratio=0.5):
    '''Loads the deepmoji dataset with split ratio 0.5.'''

    path = os.path.join(path, f"emoji_sent_race_{ratio}")
    
    x_train, z_train, y_train = load_data(os.path.join(path, "train"), size=100000, ratio=ratio)
    x_test, z_test, y_test = load_data(os.path.join(path, "test"), size=100000, ratio=ratio)
    
    return x_train, y_train, z_train, x_test, y_test, z_test



def load_synthetic(dirname):
    '''Loads the synthetic dataset with continuous latent concept.'''
    
    synthetic_dump = load_dump(os.path.join(dirname, 'data.pkl'))
    
    x_train = synthetic_dump['x_train']
    y_train = synthetic_dump['y_train']
    
    
    x_test = synthetic_dump['x_test']
    y_test = synthetic_dump['y_test']
    return x_train, y_train, y_train, x_test, y_test, y_test



def load_dump(file_path):
    """
    Load data from a .pkl file.

    Args:
        file_path (str): The path to the .pkl file.

    Returns:
        The data from the .pkl file.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def form_dataset(male_words, fem_words, neut_words):
    X, Y = [], []

    for w, v in male_words.items():
        X.append(v)
        Y.append(0)

    for w, v in fem_words.items():
        X.append(v)
        Y.append(1)
    
    for w, v in neut_words.items():
        X.append(v)
        Y.append(2)

    return np.array(X), np.array(Y)


def load_glove(data_path='data/glove/'):
    '''Loads the GloVe embeddings of gender-biased words.'''

    male_words = load_dump(os.path.join(data_path, 'male_words.pkl'))
    fem_words = load_dump(os.path.join(data_path, 'fem_words.pkl'))
    neut_words = load_dump(os.path.join(data_path, 'neut_words.pkl'))

    X, Y = form_dataset(male_words, fem_words, neut_words)
    
    X_train_dev, X_test, y_train_dev, Y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.3, random_state=0)
    X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(
        X_train_dev, y_train_dev, test_size=0.3, random_state=0)
    return X_train, Y_train, Y_train, X_test, Y_test, Y_test



def read_jigsaw_openai(path, protected_label='religion'):
    '''Loads the Jigsaw dataset with GPT-3.5 embeddings'''

    if protected_label == 'religion' or protected_label == 'gender':
        jigsaw_openai = load_dump(os.path.join(path, 
                                  f'jigsaw_{protected_label}_openai.pkl'))
        return jigsaw_openai['x_train'], jigsaw_openai['y_train'], jigsaw_openai['z_train'],\
               jigsaw_openai['x_test'], jigsaw_openai['y_test'], jigsaw_openai['z_test']


def read_crimes(path, 
                label='ViolentCrimesPerPop', 
                sensitive_attribute='racepctblack', 
                fold=1):
    '''Loads the UCI Crimes dataset.

    Protected concept: African-American population ratio
    Task: Crimes per capita
    '''

    if not os.path.isfile(os.path.join(path, 'communities.data')):
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data",
            os.path.join(path, 'communities.data'))
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names",
            os.path.join(path, 'communities.names'))

    # create names
    names = []
    with open(os.path.join(path, 'communities.names'), 'r') as file:
        for line in file:
            if line.startswith('@attribute'):
                names.append(line.split(' ')[1])

    # load data
    data = pd.read_csv(os.path.join(path, 'communities.data'), names=names, na_values=['?'])

    to_drop = ['state', 'county', 'community', 'fold', 'communityname']
    data.fillna(0, inplace=True)
    # shuffle
    data = data.sample(frac=1, replace=False).reset_index(drop=True)

    folds = data['fold'].astype(np.int)

    y = data[label].values
    to_drop += [label]

    z = data[sensitive_attribute].values
    to_drop += [sensitive_attribute]

    data.drop(to_drop + [label], axis=1, inplace=True)

    for n in data.columns:
        data[n] = (data[n] - data[n].mean()) / data[n].std()

    x = np.array(data.values)
    return x[folds != fold], y[folds != fold], z[folds != fold], x[folds == fold], y[folds == fold], z[folds == fold]
