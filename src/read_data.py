import os
import pandas as pd
import numpy as np

def read_mnist(train_ratio=0.8):
  path = os.path.join(os.path.dirname(__file__), '../datasets/digit_recognizer/')
  mnist_train = pd.read_csv(path + 'mnist_train.csv').to_numpy()
  mnist_test = pd.read_csv(path + 'mnist_test.csv').to_numpy()
  mnist = np.concatenate((mnist_train, mnist_test), axis=0)
  y = mnist[:, 0]
  X = mnist[:, 1:]
  X = X / 255
  mix_ids = np.random.permutation(X.shape[0])
  train_N = int(mnist.shape[0] * train_ratio)
  train_set, test_set = mix_ids[:train_N], mix_ids[train_N:]
  X_train, y_train = X[train_set], y[train_set]
  X_test, y_test = X[test_set], y[test_set]
  return X_train, y_train, X_test, y_test

def read_breast_cancer(train_ratio=0.8):
  path = os.path.join(os.path.dirname(__file__), '../datasets/breast_cancer_diagnostic/')
  breast_cancer = pd.read_csv(path + 'data.csv')
  breast_cancer.loc[breast_cancer['diagnosis'] == 'M', 'diagnosis'] = 1
  breast_cancer.loc[breast_cancer['diagnosis'] == 'B', 'diagnosis'] = 0
  X = breast_cancer.iloc[:, 2:-1].to_numpy()
  y = breast_cancer['diagnosis'].to_numpy()
  X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
  X = (X - X_min) / (X_max - X_min)
  mix_ids = np.random.permutation(X.shape[0])
  train_N = int(X.shape[0] * train_ratio)
  train_set, test_set = mix_ids[:train_N], mix_ids[train_N:]
  X_train, y_train = X[train_set], y[train_set]
  X_test, y_test = X[test_set], y[test_set]
  return X_train, y_train, X_test, y_test
