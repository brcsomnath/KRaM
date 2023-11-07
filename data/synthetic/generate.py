import pickle
import random
import sklearn

import numpy as np


def dump_pkl(content, filename):
    '''Dumps content into a pickle file.'''

    with open(filename, "wb") as file:
        pickle.dump(content, file)


def generate_synthetic(dim=3, N=10000):
  '''
  Generates the synthetic dataset
  with a continuous latent concept.
  '''

  samples = []
  labels = []
  cov = np.eye(dim)
  for i in range(N):
      a = random.random()
      mu = a * np.ones(dim) 
      example = np.random.multivariate_normal(mu, cov, size=1)
      samples.append(np.squeeze(example))
      labels.append(a)
  
  samples = np.array(samples)
  labels = np.array(labels)
  
  
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
      samples, labels, test_size=0.2, random_state=0)
  
  return x_train, y_train, x_test, y_test


def generate_and_dump():
    '''
    Generates the synthetic data and dumps
    it into a pickle file.
    '''

    x_train, y_train, x_test, y_test = generate_synthetic()
    
    synthetic_dump = {}
    synthetic_dump['x_train'] = x_train
    synthetic_dump['y_train'] = y_train
    synthetic_dump['x_test'] = x_test
    synthetic_dump['y_test'] = y_test

    dump_pkl(synthetic_dump, 'data.pkl')


if __name__ == '__main__':
    generate_and_dump()